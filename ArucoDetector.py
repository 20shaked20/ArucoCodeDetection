import cv2
import pandas as pd
import numpy as np
import math
import logging
import time

class ArucoDetector:
    def __init__(self, video_path, output_csv_path, output_video_path, marker_size=0.05, focal_length=600):
        self.video_path = video_path
        self.output_csv_path = output_csv_path
        self.output_video_path = output_video_path
        self.marker_size = marker_size
        self.focal_length = focal_length
        self.aruco_dict  = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self.parameters  = cv2.aruco.DetectorParameters()
        self.csv_data = []

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('ArucoDetector')

    def detect_and_draw_markers(self, frame, frame_id):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

        if ids is not None:
            for i, corner in enumerate(corners):
                cv2.aruco.drawDetectedMarkers(frame, corners)

                aruco_id = ids[i][0]
                corner = corner[0]
                top_left, top_right, bottom_right, bottom_left = corner

                dist = (self.marker_size * self.focal_length) / np.linalg.norm(top_left - top_right)
                vector = top_right - top_left
                yaw = math.degrees(math.atan2(vector[1], vector[0]))

                self.csv_data.append([frame_id, aruco_id,
                                      [top_left.tolist(), top_right.tolist(), bottom_right.tolist(), bottom_left.tolist()],
                                      dist, yaw])

                cv2.polylines(frame, [corner.astype(int)], True, (0, 255, 0), 2)
                cv2.putText(frame, str(aruco_id), tuple(top_left.astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)  # Changed color to blue
        return frame

    def process_video(self):
        self.logger.info("Starting video processing...")
        start_time = time.time()

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.logger.error("Error opening video file")
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(self.output_video_path, fourcc, fps, (frame_width, frame_height))

        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = self.detect_and_draw_markers(frame, frame_id)
            out.write(frame)
            frame_id += 1

        cap.release()
        out.release()

        end_time = time.time()
        self.save_csv()

        self.logger.info(f"Video processing completed. Total time: {end_time - start_time:.2f} seconds.")

    def save_csv(self):
        df = pd.DataFrame(self.csv_data, columns=['Frame ID', 'Aruco ID', 'Aruco 2D', 'Distance', 'Yaw'])
        df.to_csv(self.output_csv_path, index=False)
        self.logger.info(f"CSV file saved to {self.output_csv_path}")

if __name__ == "__main__":
    video_path = 'challengeB.mp4'
    output_csv_path = 'output.csv'
    output_video_path = 'output_video.avi'

    detector = ArucoDetector(video_path, output_csv_path, output_video_path)
    detector.process_video()
