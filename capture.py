
import math
import time
# import cv2
import numpy as np
import pyrealsense2 as rs
import cv2
import pandas as pd
import os

current_path = os.getcwd()

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
points = rs.points
frames = pipeline.wait_for_frames()

# Get stream profile and camera intrinsics
# profile = pipeline.get_active_profile()
# depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
# depth_intrinsics = depth_profile.get_intrinsics()
# w, h = depth_intrinsics.width, depth_intrinsics.height

# Processing blocks
pc = rs.pointcloud()
decimate = rs.decimation_filter()
decimate.set_option(rs.option.filter_magnitude, 2 ** 2)
colorizer = rs.colorizer()
file_name = 0
while True:
    userInput = input("Press key to take snapshot....")    
    if userInput == "z":
        exit
    depth = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    depth = decimate.process(depth)

    points = pc.calculate(depth)
    depth_image = np.asanyarray(depth.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    depth_colormap = np.asanyarray(colorizer.colorize(depth).get_data())
    vertecies = points.get_vertices()
    verts = np.asanyarray(vertecies)
    print("////////////////////////////////////////////////////////////////////////////")
    # for vert in verts:
    #     print(vert)
    print(verts.shape)
    # cv2.imshow("Image", depth_colormap)
    # cv2.waitKey(10)
    # data_frame = pd.DataFrame(data=verts, index = ['Row_' + str(i + 1)  
    #                     for i in range(verts.shape[0])],columns=["x", "y", "z"])
    # print(data_frame)
    # drop_series = (data_frame != 0).any(axis=1)
    # filtered_data = data_frame.loc[drop_series]

    # filtered_data.to_csv('data' + str(file_name))
    save_path = os.path.join(current_path, 'captures')
    np.savetxt(save_path + '/data' + str(file_name) + '.csv', verts, delimiter=",")
    cv2.imwrite(save_path + '/data' + str(file_name)+ '.jpg', depth_colormap)
    file_name+=1