import cv2
import numpy as np
from PIL import Image
from tensorflow.keras import models
import pyrealsense2.pyrealsense2 as rs

keyPoint = False
#Load the saved model

def main():
    model = models.load_model('new.h5')
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 480, 270, rs.format.z16, 15)
    #config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    # config.enable_stream(rs.stream.infrared, 480, 270, rs.format.y8, 15)
    pipeline.start(config)
    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            # ir_frame = frames.get_infrared_frame()
            #color_frame = frames.get_color_frame()
            if not depth_frame:
                continue


            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            # ir_image = np.asanyarray(ir_frame.get_data())
            #color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Stack both images horizontally
            # images = np.hstack((ir_image))

            # Show images
       
            # cvtColor(color_mat, color_mat, cv::COLOR_GRAY2RGB);
        
            # #Convert the captured frame into RGB
            im = Image.fromarray(depth_colormap, 'RGB')
            # im = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2RGB)
            #im = depth_colormap
            # #Resizing into 128x128 because we trained the model with this image size.
            im = im.resize((256,256))
            # img_array = np.array(im)

            # #Our keras model used a 4D tensor, (images x height x width x channel)
            # #So changing dimension 128x128x3 into 1x128x128x3 
            color_array = np.expand_dims(im, axis=0)
            # ir_array = np.expand_dims(im, axis=3)


            # #Calling the predict method on model to predict 'me' on the image
            prediction = int(model.predict_classes(color_array))

            # #if prediction is 0, which means I am missing on the image, then show the frame in gray color.
            print(prediction)
            if prediction == 0:
                print ("Chair")
                keyPoint = True
                    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            elif prediction == 1:
                print("Bed")
                keyPoint = False
            else:
                print("Couch")
                keyPoint = False
            # cv2.imshow("Capturing", frame)
            # key=cv2.waitKey(1)
            # if key == ord('q'):
            #         break
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', depth_colormap)
            cv2.waitKey(1)
    finally:
        pipeline.stop()

if __name__ == "__main__":
    main()
