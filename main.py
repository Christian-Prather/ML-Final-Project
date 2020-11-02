
import math
import time
# import cv2
import numpy as np
import pyrealsense2 as rs


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
# colorizer = rs.colorizer()
depth = frames.get_depth_frame()
depth = decimate.process(depth)

points = pc.calculate(depth)
depth_image = np.asanyarray(depth.get_data())
vertecies = points.get_vertices()
verts = np.asanyarray(vertecies)
print("////////////////////////////////////////////////////////////////////////////")
# for vert in verts:
#     print(vert)
print(verts.shape)

np.savetxt('data.csv', verts, delimiter=",")