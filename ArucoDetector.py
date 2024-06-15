import threading
import cv2
import numpy as np
import imutils
import csv
import time

class ArucoDetection:

    def __init__(self, aruco_dict, camera_matrix, dist_coeffs, marker_length):
        """Initialize the ArucoDetection class with camera parameters and ArUco dictionary."""
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
        """
        Set the image to process.
        
        Args:
            img (numpy.ndarray): The image to be processed.
        """
        with self.lock:
            self.img = img.copy()

    def preprocess_frame(self, frame):
        """
        Preprocess the frame to improve detection accuracy.
        
        Converts the frame to grayscale, applies CLAHE (Contrast Limited Adaptive Histogram Equalization),
        and blurs the image to reduce noise.

        Args:
            frame (numpy.ndarray): The input frame to be preprocessed.

        Returns:
            numpy.ndarray: The preprocessed frame.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        return blur

    def detect_aruco(self):
        """
        Detect ArUco markers in the image.
        
        Continuously checks for a new image to process and detects ArUco markers in it. The detection
        parameters are fine-tuned for better accuracy. The detected markers' corners, ids, rotation vectors,
        and translation vectors are stored.
        """
        while not self.stop_event.is_set():
            with self.lock:
                if self.img is not None:
                    img = self.img.copy()
                else:
                    continue

            image = imutils.resize(img, width=1080)  # Process at higher resolution for a better accuracy
            processed_image = self.preprocess_frame(image)
            arucoDict = cv2.aruco.getPredefinedDictionary(self.ARUCO_DICT["DICT_4X4_100"])
            arucoParams = cv2.aruco.DetectorParameters()

            # Fine-tuning the detector parameters
            arucoParams.adaptiveThreshWinSizeMin = 5
            arucoParams.adaptiveThreshWinSizeMax = 25
            arucoParams.adaptiveThreshWinSizeStep = 5
            arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            arucoParams.cornerRefinementWinSize = 10
            arucoParams.cornerRefinementMaxIterations = 100
            arucoParams.cornerRefinementMinAccuracy = 0.05

            (corners, ids, rejected) = cv2.aruco.detectMarkers(processed_image, arucoDict, parameters=arucoParams)

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

    def draw_detection(self, image, frame_id):
        """
        Draw detections on the image and collect data.
        
        Draws the detected ArUco markers and their axes on the image,
        along with the 2D corner points and the 3D pose information.

        Args:
            image (numpy.ndarray): The image on which to draw the detections.
            frame_id (int): The ID of the current frame.

        Returns:
            numpy.ndarray: The image with detections drawn on it.
        """
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

                    lookAt_point = np.array([0, 0, 1])  # Assuming camera looks along the z-axis
                    marker_vector = tvec.reshape(3)
                    lookAt_angle = float(np.degrees(np.arccos(np.dot(lookAt_point, marker_vector) / np.linalg.norm(marker_vector))))

                    # Collect data
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

        return image

    def write_to_csv(self, output_csv_path):
        """
        Write the processed data to a CSV file.

        Args:
            output_csv_path (str): The path to the output CSV file.
        """
        with open(output_csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Frame ID', 'Aruco ID', 'Aruco 2D Points', 'Distance', 'Yaw', 'Pitch', 'Roll'])
            csv_writer.writerows(self.processed_data)

    def process_video(self, video_source, output_path, output_csv_path):
        """
        Process video frames to detect ArUco codes and export the result.
        
        Reads frames from the video source, processes them to detect ArUco markers, and writes the result
        to the output path.

        Args:
            video_source (str): The path to the input video file.
            output_path (str): The path to the output video file.
            output_csv_path (str): The path to the output CSV file.
        """
        cap = cv2.VideoCapture(video_source)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for the output video
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        threading.Thread(target=self.detect_aruco, daemon=True).start()

        frame_times = []
        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            start_time = time.time()
            self.set_image_to_process(frame)
            frame = self.draw_detection(frame, frame_id)
            out.write(frame)
            end_time = time.time()

            frame_times.append(end_time - start_time)

            cv2.imshow('Aruco Detection', frame)
            if cv2.waitKey(int(300 / fps)) & 0xFF == ord('q'):
                break

            frame_id += 1

        self.stop_event.set()
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Requirement was for 'real-time' processing, so we measure the average processing time
        # In our tests, we took around ~13ms per frame
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

    # Camera calibrations using ROS (given the camera details we had)
    camera_matrix = np.array([[921.170702, 0.000000, 459.904354], [0.000000, 919.018377, 351.238301], [0.000000, 0.000000, 1.000000]])
    dist_coeffs = np.array([-0.033458, 0.105152, 0.001256, -0.006647, 0.000000])

    marker_length = 0.05  # Marker length in meters

    detection = ArucoDetection(aruco_dict, camera_matrix, dist_coeffs, marker_length)
    video_source = "challengeB.mp4" # Change this if you want a file of your own to run
    output_path = 'output_video.mp4'
    output_csv_path = 'output_data.csv'
    detection.process_video(video_source, output_path, output_csv_path)