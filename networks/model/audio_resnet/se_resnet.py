# https://amaarora.github.io/posts/2020-07-24-SeNet.html

# Modify from torchvision
# ResNeXt: Copy from https://github.com/last-one/tools/blob/master/pytorch/SE-ResNeXt/SeResNeXt.py
import torch.nn as nn
import math
import torch
import torch.nn.functional as F
import networks.model.audio_resnet.pooling_layers as pooling_layers
import torchvision
from torchvision import models


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class SEBottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, r=16):
        super(SEBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        # Add SE block
        self.se = SE_Block(planes, r)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        # Add SE operation
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, groups=1,
                 base_width=64, dilation=1, norm_layer=None, r=16):
        super(SEBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.stride = stride
        # add SE block
        self.se = SE_Block(planes, r)
        self.downsample = nn.Sequential()
        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False), nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # add SE operation
        out = self.se(out)

        identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class SEResNet(nn.Module):
    def __init__(self,
                 block,
                 num_blocks,
                 m_channels=32,
                 feat_dim=40,
                 embed_dim=128,
                 pooling_func='TSTP',
                 two_emb_layer=False):
        super(SEResNet, self).__init__()
        self.in_planes = m_channels
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.stats_dim = int(feat_dim / 8) * m_channels * 8
        self.two_emb_layer = two_emb_layer

        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, m_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, m_channels * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, m_channels * 8, num_blocks[3], stride=2)
        self.pool = getattr(pooling_layers,
                            pooling_func)(in_dim=self.stats_dim *
                                          block.expansion)
        self.pool_out_dim = self.pool.get_out_dim()
        self.seg_1 = nn.Linear(self.pool_out_dim, embed_dim)
        if self.two_emb_layer:
            self.seg_bn_1 = nn.BatchNorm1d(embed_dim, affine=False)
            self.seg_2 = nn.Linear(embed_dim, embed_dim)
        else:
            self.seg_bn_1 = nn.Identity()
            self.seg_2 = nn.Identity()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, x_len):
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = x.unsqueeze_(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        stats = self.pool(out)
        embed_a = self.seg_1(stats)
        return embed_a

    def output_size(self):
        return self.embed_dim
    
    def output_size(self):
        return self.embed_dim


def se_resnet10(feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=False):
    return SEResNet(SEBasicBlock, [1, 1, 1, 1], feat_dim=feat_dim, embed_dim=embed_dim, pooling_func=pooling_func, two_emb_layer=two_emb_layer)

def se_resnet18(feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=False):
    return SEResNet(SEBasicBlock, [2, 2, 2, 2], feat_dim=feat_dim, embed_dim=embed_dim, pooling_func=pooling_func, two_emb_layer=two_emb_layer)


def se_resnet34(feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=False):
    return SEResNet(SEBasicBlock, [3, 4, 6, 3], feat_dim=feat_dim, embed_dim=embed_dim, pooling_func=pooling_func, two_emb_layer=two_emb_layer)



def get_seresnet(resnum, feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=False):
    if resnum == 10:
        model = se_resnet10(feat_dim=feat_dim,  embed_dim=embed_dim, pooling_func=pooling_func, two_emb_layer=two_emb_layer)
    elif resnum == 18:
        model = se_resnet18(feat_dim=feat_dim,  embed_dim=embed_dim, pooling_func=pooling_func, two_emb_layer=two_emb_layer)
    elif resnum == 34:
        model = se_resnet34(feat_dim=feat_dim,  embed_dim=embed_dim, pooling_func=pooling_func, two_emb_layer=two_emb_layer)
    else:
        raise ValueError("Unsupported resnum: {}".format(resnum))
    return model


if __name__ == '__main__':
    x = torch.zeros(8, 200, 80)
    x_len = torch.tensor([200])
    model = get_seresnet(resnum=18, feat_dim=80, embed_dim=256)
    model.eval()
    out = model(x, x_len)
    print(out[-1].size())

    num_params = sum(p.numel() for p in model.parameters())
    print("{} M".format(num_params / 1e6))