import os
import cv2
import numpy as np
from numpy import *
from flow_vis import flow_to_color


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


# 光流可视化
def draw_optical_flow(flow_data, output_path):
    height, width = flow_data[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

    for flow_frame in flow_data:
        # Convert flow to color image
        flow_color = flow_to_color(flow_frame, convert_to_bgr=True)
        out.write(flow_color)

    out.release()
    cv2.destroyAllWindows()


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

if __name__ == "__main__":
  # video path https://www.pexels.com/search/videos/face/
  # flow https://blog.csdn.net/yangchuangyc/article/details/124946688
  video_file = "3762907-hd_1920_1080_25fps.mp4"
  for method in ['SparseToDense', 'Farneback', 'PCAFlow', 'DeepFlow', 'SF', 'DualTVL1', 'DIS']:
    print(method)
    # flow_data = calculate_optical_flow(video_file, method=method)
    # np.save(f"optical_flow_{method}.npy", flow_data)
    flow_data = np.load(f"optical_flow_{method}.npy")
    draw_optical_flow(flow_data, f"optical_flow_{method}.avi")
    draw_optical_flow_xy(flow_data, f"optical_flow_{method}")






