import cv2
import pandas as pd
import numpy as np
import math
import logging
import time

class ArucoDetector:
    def __init__(self, video_path, output_csv_path, output_video_path, camera_matrix, distortion, marker_size=0.05):
        """
        Initialize the ArucoDetector with the video path, output CSV path, output video path,
        camera matrix, distortion coefficients, and marker size.

        :param video_path: Path to the input video file.
        :param output_csv_path: Path to save the output CSV file.
        :param output_video_path: Path to save the output video file.
        :param camera_matrix: Camera matrix from calibration.
        :param distortion: Distortion coefficients from calibration.
        :param marker_size: Size of the ArUco marker in meters.
        """

        self.video_path = video_path
        self.output_csv_path = output_csv_path
        self.output_video_path = output_video_path
        self.camera_matrix = camera_matrix
        self.distortion = distortion
        self.marker_size = marker_size
        self.aruco_dict  = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self.parameters  = cv2.aruco.DetectorParameters()
        self.csv_data = []

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('ArucoDetector')

    def detect_and_draw_markers(self, frame, frame_id):
        """
        Detects ArUco markers in a given frame, draws bounding boxes around them, and logs their information.

        :param frame: The video frame in which to detect ArUco markers.
        :param frame_id: The ID of the current frame being processed.
        :return: The frame with detected ArUco markers drawn.
        """

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

        if ids is not None:
            for i, corner in enumerate(corners):
                cv2.aruco.drawDetectedMarkers(frame, corners)

                aruco_id = ids[i][0]
                corner = corner[0]
                top_left, top_right, bottom_right, bottom_left = corner

                # Calculate distance from the camera to the marker
                dist = (self.marker_size * self.camera_matrix[0, 0]) / np.linalg.norm(top_left - top_right)
                # Calculate yaw (rotation around the vertical axis)
                vector = top_right - top_left
                yaw = math.degrees(math.atan2(vector[1], vector[0]))

                # Add marker information to CSV data
                self.csv_data.append([frame_id, aruco_id,
                                      [top_left.tolist(), top_right.tolist(), bottom_right.tolist(), bottom_left.tolist()],
                                      dist, yaw])

                # Draw the bounding box and marker ID on the frame
                cv2.polylines(frame, [corner.astype(int)], True, (0, 255, 0), 2)
                cv2.putText(frame, str(aruco_id), tuple(top_left.astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)  # Changed color to blue
        return frame

    def process_video(self):
        """
        Processes the input video frame by frame, detects and draws ArUco markers, and writes the output video.
        Logs the start and end of processing and the total time taken.
        """

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

        Time = end_time-start_time

        self.logger.info(f"Video processing completed. Total time: {Time:.2f} seconds.")
        self.logger.info(f"Each Frame on average took to process: {(Time/ frame_id):.2f} seconds")


    def save_csv(self):
        """
        Saves the collected ArUco marker information to a CSV file.
        Logs the path to the saved CSV file.
        """
        
        df = pd.DataFrame(self.csv_data, columns=['Frame ID', 'Aruco ID', 'Aruco 2D', 'Distance', 'Yaw'])
        df.to_csv(self.output_csv_path, index=False)
        self.logger.info(f"CSV file saved to {self.output_csv_path}")

if __name__ == "__main__":
    video_path = 'challengeB2.mp4'
    output_csv_path = 'output.csv'
    output_video_path = 'output_video.avi'

    #camera calibrations as sent in the assignment
    camera_matrix = np.array([[921.170702, 0.000000, 459.904354],
                              [0.000000, 919.018377, 351.238301],
                              [0.000000, 0.000000, 1.000000]])
    distortion = np.array([-0.033458, 0.105152, 0.001256, -0.006647, 0.000000])

    detector = ArucoDetector(video_path, output_csv_path, output_video_path, camera_matrix, distortion)
    detector.process_video()
