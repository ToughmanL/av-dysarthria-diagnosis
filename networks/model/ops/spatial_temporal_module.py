import torch
import torch.nn as nn
import torch.nn.functional as F

# 空间池化
class SimpleSpatialModule(nn.Module):
    def __init__(self, spatial_type='avg', spatial_size=7):
        super(SimpleSpatialModule, self).__init__()
        assert spatial_type in ['avg']
        self.spatial_type = spatial_type
        self.spatial_size = spatial_size if not isinstance(spatial_size, int) else (spatial_size, spatial_size)
        if self.spatial_type == 'avg':
            self.op = nn.AvgPool2d(self.spatial_size, stride=1, padding=0)

    def init_weights(self):
        pass

    def forward(self, input):
        return self.op(input)

# 时空池化
class SimpleSpatialTemporalModule(nn.Module):
    def __init__(self, spatial_type='avg', spatial_size=7, temporal_size=1):
        super(SimpleSpatialTemporalModule, self).__init__()
        assert spatial_type in ['identity', 'avg', 'max']
        self.spatial_type = spatial_type
        self.spatial_size = spatial_size
        if spatial_size != -1:
            self.spatial_size = (spatial_size, spatial_size)

        self.temporal_size = temporal_size
        assert not (self.spatial_size == -1) ^ (self.temporal_size == -1)

        if self.temporal_size == -1 and self.spatial_size == -1:
            self.pool_size = (1, 1, 1)
            if self.spatial_type == 'avg':
                self.pool_func = nn.AdaptiveAvgPool3d(self.pool_size)
            if self.spatial_type == 'max':
                self.pool_func = nn.AdaptiveMaxPool3d(self.pool_size)
        else:
            self.pool_size = (self.temporal_size, ) + self.spatial_size
            if self.spatial_type == 'avg':
                self.pool_func = nn.AvgPool3d(self.pool_size, stride=1, padding=0)
            if self.spatial_type == 'max':
                self.pool_func = nn.MaxPool3d(self.pool_size, stride=1, padding=0)

    def init_weights(self):
        pass

    def forward(self, input):
        if self.spatial_type == 'identity':
            return input
        else:
            return self.pool_func(input)


# 快慢时空池化
class SlowFastSpatialTemporalModule(nn.Module):
    def __init__(self, adaptive_pool=True, spatial_type='avg', spatial_size=1, temporal_size=1):
        super(SlowFastSpatialTemporalModule, self).__init__()

        self.adaptive_pool = adaptive_pool
        assert spatial_type in ['avg']
        self.spatial_type = spatial_type

        self.spatial_size = spatial_size if not isinstance(spatial_size, int) else (spatial_size, spatial_size)
        self.temporal_size = temporal_size
        self.pool_size = (self.temporal_size, ) + self.spatial_size

        if self.adaptive_pool:
            if self.spatial_type == 'avg':
                self.op = nn.AdaptiveAvgPool3d(self.pool_size)
        else:
            raise NotImplementedError


    def init_weights(self):
        pass

    def forward(self, input):
        x_slow, x_fast = input
        x_slow = self.op(x_slow)
        x_fast = self.op(x_fast)
        return torch.cat((x_slow, x_fast), dim=1)