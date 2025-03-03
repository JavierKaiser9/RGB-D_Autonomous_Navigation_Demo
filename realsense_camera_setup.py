import numpy as np
import cv2
import pyrealsense2 as rs
import os


class Camera:
    """SETTINGS FOR USING THE REALSENSE RGB-D 415 CAMERA"""

    # Constructor for the RealSense Depth Camera 415
    def __init__(self, X=640, framerate=30):
        Y = self.create_ratio(X)
        self.pipeline, self.config = self.setup_pipeline(X, Y, framerate)
        self.pipeline.start(self.config)

    # Camera Pipeline Setup
    def setup_pipeline(self, X, Y, framerate):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, X, Y, rs.format.z16, framerate)
        config.enable_stream(rs.stream.color, X, Y, rs.format.bgr8, framerate)
        return pipeline, config

    # Settings the correct dimensions for the camera
    def create_ratio(self, X):
        dict = {
            424: 240,
            640: 480,
            1280: 720
        }
        return dict.get(X, 480)  # Default to 480 if X not in the dictionary

    # Frames for both RGB and Depth cameras
    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        return np.asanyarray(depth_frame.get_data()), np.asanyarray(color_frame.get_data())

    # Resize with opencv
    def resize(self, src, height, width):
        return cv2.resize(src, (height, width))

    # Press button 'ESC' to exit the program
    def final_camera(self):
        cv2.destroyAllWindows()
        self.pipeline.stop()


if __name__ == "__main__":
    cam = Camera(640, 30)
    while True:

        # Display RGB-D camera for the user
        depth_img, color_img = cam.get_frame()
        depth_map_jet = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow('depth', depth_map_jet)
        cv2.imshow('color', color_img)

        # Check if a key was pressed
        key = cv2.waitKey(1)

        # Press 'Esc' key to exit
        if key == 27:
            cam.final_camera()
            break
