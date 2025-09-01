# Code for paper:
# [Title]  - "PAN: Towards Fast Action Recognition via Learning Persistence of Appearance"
# [Author] - Can Zhang, Yuexian Zou, Guang Chen, Lei Gan
# [Github] - https://github.com/zhang-can/PAN-PyTorch

import torch
from torch import nn
import math
import cv2
import numpy as np

class PA1(nn.Module):
    def __init__(self, n_length=4):
        super(PA, self).__init__()
        self.shallow_conv = nn.Conv2d(3,8,7,1,3)
        self.n_length = n_length
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.001)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        (n, c, h, w) = x.shape
        x = x.view((-1, 3) + x.size()[-2:])
        x = self.shallow_conv(x)
        distances = []
        for i in range(n-1):
            curr_frame = x[i,:,:,:].view(-1, 8)
            last_frame = x[i+1,:,:,:].view(-1, 8)
            distance = nn.PairwiseDistance(p=2)(curr_frame, last_frame)
            distances.append(distance.view(-1, h, w))
        PA = torch.stack(distances, dim=1)
        PA = PA.view(-1, 1*(n-1), h, w)
        return PA

    def forward_4(self, x):
        h, w = x.size(-2), x.size(-1)
        x = x.view((-1, 3) + x.size()[-2:])
        x = self.shallow_conv(x)
        distances = []
        for i in range(self.n_length-1):
            curr_frame = x[i,:,:,:].view(-1, 8)
            last_frame = x[i+1,:,:,:].view(-1, 8)
            distance = nn.PairwiseDistance(p=2)(curr_frame, last_frame)
            distances.append(distance.view(-1, h, w))
        PA = torch.stack(distances, dim=1)
        return PA

    def forward_ori(self, x):
        h, w = x.size(-2), x.size(-1)
        x = x.view((-1, 3) + x.size()[-2:])
        x = self.shallow_conv(x)
        x = x.view(-1, self.n_length, x.size(-3), x.size(-2)*x.size(-1))
        for i in range(self.n_length-1):
            d_i = nn.PairwiseDistance(p=2)(x[:,i,:,:], x[:,i+1,:,:]).unsqueeze(1)
            d = d_i if i == 0 else torch.cat((d, d_i), 1)
        PA = d.view(-1, 1*(self.n_length-1), h, w)
        return PA

class PA(nn.Module):
    def __init__(self, n_length):
        super(PA, self).__init__()
        self.shallow_conv = nn.Conv2d(3,3,3,1,1)
        self.n_length = n_length
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.001)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        #输入x为[N*T*m, 3, H, W]
        h, w = x.size(-2), x.size(-1)
        x = x.view(-1, self.n_length, x.size(-3), x.size(-2)*x.size(-1)).unsqueeze(-1)
        print(x[:,0,:,:].shape)
        for i in range(self.n_length-1):
            d_i = nn.PairwiseDistance(p=2)(x[:,i,:,:], x[:,i+1,:,:]).unsqueeze(1) # 公式4
            d = d_i if i == 0 else torch.cat((d, d_i), 1)
        x = d.unsqueeze(-1)
        print(x.shape)
        for i in range(self.n_length-2):
            if i == 0:
                p = x[:, 0, :, :]
            else:
                p = nn.PairwiseDistance(p=2)(x[:,i-1,:,:], p).unsqueeze(1)
                p = p.transpose(2,3)
        PA_data = torch.norm(d, p = 2, dim = 2)   #公式5
        PA_data_str = torch.norm(p.transpose(2,3), p = 2, dim = 2)
        PA_data = PA_data.view(-1, 1*(self.n_length-1), h, w)
        PA_data_str = PA_data_str.view(-1, 1, h, w)
        print(PA_data.shape)
        print(PA_data_str.shape)
        return PA_data.cpu().numpy().copy(), PA_data_str.cpu().numpy().copy()

def get_paflow(x):
    #输入x为[N*T*m, 3, H, W]
    b, s, t, c, h, w = x.size()
    x = x.view(-1, t, c, h*w).unsqueeze(-1)
    for i in range(t-1):
        d_i = nn.PairwiseDistance(p=2)(x[:,i,:,:,:], x[:,i+1,:,:,:]).unsqueeze(1) # 公式4
        d = d_i if i == 0 else torch.cat((d, d_i), 1)
    PA_data = torch.norm(d, p = 2, dim = 2)   #公式5
    PA_data = PA_data.view(-1, 1*(t-1), h, w)
    return PA_data.cpu().numpy().copy()

class VAP(nn.Module):
    def __init__(self, n_segment, feature_dim, num_class, dropout_ratio):
        super(VAP, self).__init__()
        VAP_level = int(math.log(n_segment, 2))
        print("=> Using {}-level VAP".format(VAP_level))
        self.n_segment = n_segment
        self.VAP_level = VAP_level
        total_timescale = 0
        for i in range(VAP_level):
           timescale = 2**i
           total_timescale += timescale
           setattr(self, "VAP_{}".format(timescale), nn.MaxPool3d((n_segment//timescale,1,1),1,0,(timescale,1,1)))
        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.TES = nn.Sequential(
            nn.Linear(total_timescale, total_timescale*4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(total_timescale*4, total_timescale, bias=False)
        )
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.pred = nn.Linear(feature_dim, num_class)
        
        # fc init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.001)
                if hasattr(m.bias, 'data'):
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        _, d = x.size()
        x = x.view(-1, self.n_segment, d, 1, 1).permute(0,2,1,3,4)
        x = torch.cat(tuple([getattr(self, "VAP_{}".format(2**i))(x) for i in range(self.VAP_level)]), 2).squeeze(3).squeeze(3).permute(0,2,1)
        w = self.GAP(x).squeeze(2)
        w = self.softmax(self.TES(w))
        x = x * w.unsqueeze(2)
        x = x.sum(dim=1)
        x = self.dropout(x)
        x = self.pred(x.view(-1,d))
        return x


# 读取视频文件, 返回视频数据
def read_video(video_path):
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 将图像从默认的BGR格式转换为RGB格式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 将图像转换为PyTorch张量，并转换通道顺序为[C, H, W]
        frame_tensor = torch.from_numpy(frame_rgb.transpose((2, 0, 1))).float()
        frames.append(frame_tensor)
    cap.release()
    frames_tensor = torch.stack(frames)
    return frames_tensor

# numpy data save to avi grey video
def save_video(data, data_path):
    data = data.squeeze()
    # data = np.clip(data, 0, 255)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    height, width = data.shape[1], data.shape[2]
    out = cv2.VideoWriter(data_path, fourcc, 25.0, (width, height), isColor=False)
    for frame in data:
        frame = cv2.convertScaleAbs(frame)
        out.write(frame)
    out.release()

# test PA
if __name__ == '__main__':
    # video_path = "data/3762907-hd_1920_1080_25fps.mp4"
    video_path = "workspace/av-dysarthria-diagnosis/egs/msdm/data/MSDM/crop_video/seg_data/S_M_00034_G1_task3_2_S00011.avi"
    frames = read_video(video_path)
    frames = frames
    (n, c, h, w) = frames.shape
    # pa = PA(n)
    frames = frames.unsqueeze(0)
    frames = frames.unsqueeze(0)
    batch_frames = frames.repeat(1, 1, 1, 1, 1, 1)
    # PA_data, PA_data_str = pa(frames)
    PA_data = get_paflow(batch_frames)
    save_video(PA_data, "PA_data.avi")
