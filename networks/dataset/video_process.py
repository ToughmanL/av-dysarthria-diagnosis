import os
import cv2
import torchvision.transforms.functional as F
import numpy as np
from numpy import *
from flow_vis import flow_to_color
import torch
import torch.nn as nn

# read video
def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
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

# video normalization
def normalize_video(video_data):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    video_data = video_data / 255.0
    for frame in video_data:
        frame.sub_(mean[:, None, None]).div_(std[:, None, None])
    return video_data

# crop lip
def crop_lip(input_tensor, crop_size):
    h, w, d = input_tensor.shape
    crop_w, crop_d = crop_size
    start_w = (w - crop_w) // 2
    end_w = start_w + crop_w
    start_d = (d - crop_d) // 2
    end_d = start_d + crop_d
    cropped_tensor = input_tensor[:, start_w:end_w, start_d:end_d]
    return cropped_tensor

def read_and_convert_to_grayscale(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
        break
      # Convert to grayscale
      gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      # Convert to PyTorch tensor
      tensor_frame = F.to_tensor(gray_frame)
      # Normalize the tensor (optional)
    #   tensor_frame = F.normalize(tensor_frame, [0.5], [0.5])
      if tensor_frame.shape[2] > 80:
        tensor_frame = crop_lip(tensor_frame, (80, 80))
      frames.append(tensor_frame)
    cap.release()
    if len(frames) == 0:
      tensor_frame = torch.rand(1,80,80)
      frames.append(tensor_frame)
      frames.append(tensor_frame)
    return torch.stack(frames)

# 计算光流
def calculate_optical_flow(video_path, method='DeepFlow'):
    cap = cv2.VideoCapture(video_path)
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    flow_data = []

    while True:
        ret, frame2 = cap.read()
        if not ret:
            break
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        if method == 'DeepFlow':
            flow = cv2.optflow.createOptFlow_DeepFlow()
            flow_frame = flow.calc(prvs, next, None)
        elif method == 'SparseToDense':
            flow_frame = cv2.optflow.calcOpticalFlowSparseToDense(prvs, next)
        elif method == 'Farneback':
            flow_frame = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        elif method == 'PCAFlow':
            flow = cv2.optflow.createOptFlow_PCAFlow()
            flow_frame = flow.calc(prvs, next, None)
        elif method == 'SF':
            flow_frame = cv2.optflow.calcOpticalFlowSF(prvs, next, 3, 5, 5)
        elif method == 'DualTVL1':
            flow = cv2.optflow.DualTVL1OpticalFlow_create()
            flow_frame = flow.calc(prvs, next, None)
        else:
            raise ValueError("Invalid method provided for optical flow calculation.")
        flow_data.append(flow_frame)
        prvs = next

    cap.release()
    cv2.destroyAllWindows()

    return flow_data


def compute_paflow(x):
    #输入x为[N*T*m, 3, H, W]
    b, t, c, h, w = x.size()
    x = x.view(-1, t, c, h*w).unsqueeze(-1)
    for i in range(t-1):
        d_i = nn.PairwiseDistance(p=2)(x[:,i,:,:,:], x[:,i+1,:,:,:]).unsqueeze(1) # 公式4
        d = d_i if i == 0 else torch.cat((d, d_i), 1)
    PA_data = torch.norm(d, p = 2, dim = 2)   #公式5
    PA_data = PA_data.view(-1, 1*(t-1), h, w)
    return PA_data


# 光流可视化
# def draw_optical_flow(flow_data, output_path):
#     height, width = flow_data[0].shape[:2]
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

#     for flow_frame in flow_data:
#         # Convert flow to color image
#         flow_color = flow_to_color(flow_frame, convert_to_bgr=True)
#         out.write(flow_color)
#     out.release()

# 光流可视化
def draw_optical_flow(flow_data, output_path):
    height, width = flow_data[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height), isColor=False)

    for flow_frame in flow_data:
        out.write(flow_frame)
    out.release()


def draw_optical_flow_xy(flow_data, output_path_prefix):
    height, width = flow_data[0].shape[:2]

    # Create VideoWriters for x and y directions
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # 使用 MJPG 编解码器
    out_x = cv2.VideoWriter(output_path_prefix + '_x.avi', fourcc, 20.0, (width, height))
    out_y = cv2.VideoWriter(output_path_prefix + '_y.avi', fourcc, 20.0, (width, height))

    for flow_frame in flow_data:
        # Split flow into x and y components
        flow_x = flow_frame[:, :, 0]
        flow_y = flow_frame[:, :, 1]

        # Convert flow to color images
        flow_color_x = flow_to_color(flow_x, convert_to_bgr=True)
        flow_color_y = flow_to_color(flow_y, convert_to_bgr=True)

        # Write frames to videos
        out_x.write(flow_color_x)
        out_y.write(flow_color_y)

    # Release VideoWriters
    out_x.release()
    out_y.release()
    cv2.destroyAllWindows()


def test_paflow(video_path, paflow_path):
    frames_tensor = read_and_convert_to_grayscale(video_path)
    patlow = compute_paflow(frames_tensor.unsqueeze(0))
    # patlow = frames_tensor
    patlow = patlow.squeeze(0)
    patlow = patlow.detach().numpy()
    patlow = (patlow - patlow.min()) / (patlow.max() - patlow.min()) * 255
    patlow = patlow.astype(np.uint8)
    draw_optical_flow(patlow, paflow_path)


if __name__ == "__main__":

  video_path = 'workspace/av-dysarthria-diagnosis/egs/msdm/data/MSDM/crop_video/seg_data/S_M_00078_G3_task8_1_S00001.avi'
  paflow_path = 'S_M_00078_G3_task8_1_S00001.paflow.avi'
  test_paflow(video_path, paflow_path)
  exit(-1)

  video_file = "workspace/av-dysarthria-diagnosis/egs/msdm/data/MSDM/crop_video/seg_data/S_M_00034_G1_task3_2_S00011.avi"
  for method in ['SparseToDense', 'Farneback', 'PCAFlow', 'DeepFlow', 'SF', 'DualTVL1', 'DIS', ]:
    print(method)
    flow_data = calculate_optical_flow(video_file, method=method)
    # np.save(f"optical_flow_{method}.npy", flow_data)
    # flow_data = np.load(f"optical_flow_{method}.npy")
    draw_optical_flow(flow_data, f"optical_flow_{method}.avi")
    # draw_optical_flow_xy(flow_data, f"optical_flow_{method}")






