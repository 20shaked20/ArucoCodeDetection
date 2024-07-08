import cv2
import numpy as np
import math

class ArucoTracker:
    def __init__(self, target_image_path, marker_size=0.05):
        self.marker_size = marker_size

        # Camera matrix and distortion coefficients (example values)
        self.camera_matrix = np.array([[921.170702, 0.000000, 459.904354],
                                       [0.000000, 919.018377, 351.238301],
                                       [0.000000, 0.000000, 1.000000]])
        self.distortion_coeffs = np.array([-0.033458, 0.105152, 0.001256, -0.006647, 0.000000])

        # Load target image and detect markers
        self.target_image = cv2.imread(target_image_path)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self.parameters = cv2.aruco.DetectorParameters()

        self.target_corners, self.target_ids, self.target_rvecs, self.target_tvecs = self.detect_markers(self.target_image)

    def detect_markers(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_size, self.camera_matrix, self.distortion_coeffs)
            return corners, ids, rvecs, tvecs
        else:
            return corners, ids, None, None

    def calculate_movement(self, live_rvecs, live_tvecs):
        if live_rvecs is None or live_tvecs is None:
            return []

        command = []

        target_tvec = self.target_tvecs[0][0]
        live_tvec = live_tvecs[0][0]

        # Calculate distance differences
        diff = target_tvec - live_tvec
        distance = np.linalg.norm(diff)

        # Check proximity
        if distance < 0.05:
            command.append("Position aligned")
        else:
            if diff[0] > 0.05:
                command.append("move right")
            elif diff[0] < -0.05:
                command.append("move left")
            elif diff[1] > 0.05:
                command.append("move down")
            elif diff[1] < -0.05:
                command.append("move up")
            elif diff[2] > 0.05:
                command.append("move forward")
            elif diff[2] < -0.05:
                command.append("move backward")

        # Calculate rotation (yaw) differences
        target_rvec = self.target_rvecs[0][0]
        live_rvec = live_rvecs[0][0]

        target_rotation_matrix, _ = cv2.Rodrigues(target_rvec)
        live_rotation_matrix, _ = cv2.Rodrigues(live_rvec)

        target_yaw = np.arctan2(target_rotation_matrix[1, 0], target_rotation_matrix[0, 0])
        live_yaw = np.arctan2(live_rotation_matrix[1, 0], live_rotation_matrix[0, 0])

        yaw_diff_degrees = np.degrees(target_yaw - live_yaw)

        if yaw_diff_degrees > 10:
            command.append("turn right")
        elif yaw_diff_degrees < -10:
            command.append("turn left")

        return command

    def track(self):
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            live_corners, live_ids, live_rvecs, live_tvecs = self.detect_markers(frame)

            commands = self.calculate_movement(live_rvecs, live_tvecs)

            # Draw detected markers and their IDs
            if live_ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, live_corners, live_ids)

            # Display movement commands on the frame
            if commands:
                cv2.putText(frame, commands[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    target_image_path = 'target_frame_home.jpg'  # Path to the target frame image
    tracker = ArucoTracker(target_image_path)
    tracker.track()
