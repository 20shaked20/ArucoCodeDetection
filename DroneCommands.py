import threading
import cv2
import numpy as np
import imutils
import csv
import pandas as pd
import time

class ArucoDetection:

    def __init__(self, aruco_dict, camera_matrix, dist_coeffs, marker_length):
        self.img = None
        self.ARUCO_DICT = aruco_dict

        self.camera_matrix = np.array(camera_matrix, dtype=np.float32)
        self.dist_coeffs = np.array(dist_coeffs, dtype=np.float32)
        self.marker_length = marker_length

        self.corners = []
        self.ids = []
        self.rvecs = []
        self.tvecs = []

        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        self.processed_data = []

    def set_image_to_process(self, img):
        with self.lock:
            self.img = img.copy()

    def preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        return blur

    def detect_aruco(self):
        while not self.stop_event.is_set():
            with self.lock:
                if self.img is not None:
                    img = self.img.copy()
                else:
                    continue

            image = imutils.resize(img, width=1080)
            processed_image = self.preprocess_frame(image)
            arucoDict = cv2.aruco.getPredefinedDictionary(self.ARUCO_DICT["DICT_4X4_100"])
            arucoParams = cv2.aruco.DetectorParameters()
            
            arucoParams.adaptiveThreshWinSizeMin = 3
            arucoParams.adaptiveThreshWinSizeMax = 23
            arucoParams.adaptiveThreshWinSizeStep = 5
            arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            arucoParams.cornerRefinementWinSize = 10
            arucoParams.cornerRefinementMaxIterations = 30
            arucoParams.cornerRefinementMinAccuracy = 0.01

            corners, ids, rejected = cv2.aruco.detectMarkers(processed_image, arucoDict, parameters=arucoParams)

            if ids is not None:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_length, self.camera_matrix, self.dist_coeffs)
            else:
                rvecs, tvecs = None, None

            with self.lock:
                if ids is not None:
                    self.corners = corners
                    self.ids = ids
                    self.rvecs = rvecs
                    self.tvecs = tvecs
                else:
                    self.corners = []
                    self.ids = []
                    self.rvecs = []
                    self.tvecs = []

    def get_movement_command(self, target_frame_data, current_frame_data):
        position_diff = np.mean(eval(current_frame_data['Aruco 2D Points']), axis=0) - np.mean(eval(target_frame_data['Aruco 2D Points']), axis=0)
        distance_diff = current_frame_data['Distance'] - target_frame_data['Distance']
        yaw_diff = current_frame_data['Yaw'] - target_frame_data['Yaw']
        
        command = ""
        if position_diff[1] > 10:
            command += "down "
        elif position_diff[1] < -10:
            command += "up "
        if position_diff[0] > 10:
            command += "left "
        elif position_diff[0] < -10:
            command += "right "
        if distance_diff > 0.1:
            command += "backward "
        elif distance_diff < -0.1:
            command += "forward "
        if yaw_diff > 10:
            command += "turn-left"
        elif yaw_diff < -10:
            command += "turn-right"
        
        return command.strip()

    def draw_detection(self, image, frame_id, target_frame_data):
        with self.lock:
            if self.ids is not None and self.corners is not None:
                for corner, id_, rvec, tvec in zip(self.corners, self.ids, self.rvecs, self.tvecs):
                    cv2.aruco.drawDetectedMarkers(image, [corner], id_)

                    for point in corner[0]:
                        cv2.circle(image, tuple(point.astype(int)), 5, (0, 255, 0), -1)

                    distance = np.linalg.norm(tvec)
                    rmat, _ = cv2.Rodrigues(rvec)
                    rmat = np.array(rmat)
                    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(np.hstack((rmat, tvec.reshape(3, 1))))
                    yaw, pitch, roll = [(float(np.degrees(angle.item())) % 360) for angle in euler_angles]

                    lookAt_point = np.array([0, 0, 1])
                    marker_vector = tvec.reshape(3)
                    lookAt_angle = float(np.degrees(np.arccos(np.dot(lookAt_point, marker_vector) / np.linalg.norm(marker_vector))))

                    aruco_2d_points = [
                        tuple(corner[0][0].astype(int)),
                        tuple(corner[0][1].astype(int)),
                        tuple(corner[0][2].astype(int)),
                        tuple(corner[0][3].astype(int))
                    ]
                    self.processed_data.append([frame_id, int(id_[0]), aruco_2d_points, distance, yaw, pitch, roll])

                    info_text = (f"Dist: {distance:.2f}m Yaw: {yaw:.2f} "
                                f"Pitch: {pitch:.2f} Roll: {roll:.2f} LookAt: {lookAt_angle:.2f}")
                    cv2.putText(image, info_text, (int(corner[0][0][0]), int(corner[0][0][1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    current_frame_data = {
                        'Aruco 2D Points': str(corner[0].tolist()),
                        'Distance': distance,
                        'Yaw': yaw,
                        'Pitch': pitch,
                        'Roll': roll
                    }
                    command = self.get_movement_command(target_frame_data, current_frame_data)
                    cv2.putText(image, f"Command: {command}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Print the command to the CMD
                    print(f"Frame {frame_id}, Aruco ID {id_[0]}: {command}")

        return image

    def write_to_csv(self, output_csv_path):
        with open(output_csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Frame ID', 'Aruco ID', 'Aruco 2D Points', 'Distance', 'Yaw', 'Pitch', 'Roll'])
            csv_writer.writerows(self.processed_data)

    def process_video(self, video_source, output_path, output_csv_path, target_frame_id):
        cap = cv2.VideoCapture(video_source)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        threading.Thread(target=self.detect_aruco, daemon=True).start()

        frame_times = []
        frame_id = 0

        data = pd.read_csv(output_csv_path)
        target_frame_data = data[data['Frame ID'] == target_frame_id].iloc[0]

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            start_time = time.time()
            self.set_image_to_process(frame)
            frame = self.draw_detection(frame, frame_id, target_frame_data)
            out.write(frame)
            end_time = time.time()

            frame_times.append(end_time - start_time)

            cv2.imshow('Aruco Detection', frame)
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                break

            frame_id += 1

        self.stop_event.set()
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        average_time_per_frame = sum(frame_times) / len(frame_times)
        print(f"Average time to process each frame: {average_time_per_frame:.4f} seconds")

        self.write_to_csv(output_csv_path)

if __name__ == "__main__":
    aruco_dict = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000
    }

    camera_matrix = [[921.170702, 0.000000, 459.904354], [0.000000, 919.018377, 351.238301], [0.000000, 0.000000, 1.000000]]
    dist_coeffs = [-0.033458, 0.105152, 0.001256, -0.006647, 0.000000]

    marker_length = 0.05  # Marker length in meters

    detection = ArucoDetection(aruco_dict, camera_matrix, dist_coeffs, marker_length)
    video_source = 0  # Use the webcam (or provide a video file path)
    output_path = 'output_video.mp4'
    output_csv_path = 'output_data.csv'
    
    # The target frame ID for the movement commands
    target_frame_id = 317

    detection.process_video(video_source, output_path, output_csv_path, target_frame_id)
