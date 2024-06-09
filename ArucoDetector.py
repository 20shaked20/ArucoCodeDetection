import threading
from typing import List
import cv2
import numpy as np
import imutils

class ArucoDetection:

    def __init__(self, aruco_dict):
        self.img = None
        self.ARUCO_DICT = aruco_dict
        self.corners = []
        self.ids = []
        self.contours = []

        self.lock = threading.Lock()
        self.stop_event = threading.Event()

    def set_image_to_process(self, img):
        """Set the image to process."""
        with self.lock:
            self.img = img.copy()
    
    def detect_aruco(self):
        """Detect ArUco markers in the image."""
        while not self.stop_event.is_set():
            with self.lock:
                if self.img is not None:
                    img = self.img.copy()
                else:
                    continue

            image = imutils.resize(img, width=720)
            arucoDict = cv2.aruco.getPredefinedDictionary(self.ARUCO_DICT["DICT_4X4_100"])
            arucoParams = cv2.aruco.DetectorParameters()
            (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

            with self.lock:
                if ids is not None:
                    self.corners = corners
                    self.ids = ids
                else:
                    self.corners = []
                    self.ids = []

    def draw_detection(self, image):
        """Draw detections on the image."""
        with self.lock:
            if self.ids is not None and self.corners is not None:
                for corner, id_ in zip(self.corners, self.ids):
                    cv2.aruco.drawDetectedMarkers(image, [corner], id_)
        return image

    def process_video(self, video_source, output_path):
        """Process video frames to detect ArUco codes and export the result."""
        cap = cv2.VideoCapture(video_source)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for the output video
        out = None

        threading.Thread(target=self.detect_aruco, daemon=True).start()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if out is None:
                # Initialize video writer with the same dimensions as the input frame
                height, width = frame.shape[:2]
                out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

            self.set_image_to_process(frame)
            frame = self.draw_detection(frame)
            out.write(frame)

            cv2.imshow('Aruco Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.stop_event.set()
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    aruco_dict = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000
    }

    detection = ArucoDetection(aruco_dict)
    video_source = "ArucoCodeDetection\challengeB.mp4"  # Change this to your video source (e.g., file path or camera index)
    output_path = 'output_video.mp4'  # Change this to your desired output path
    detection.process_video(video_source, output_path)
