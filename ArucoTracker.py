import cv2
import numpy as np
import math

class ArucoTracker:
    def __init__(self, target_image_path, marker_size=0.05):
        self.marker_size = marker_size

        # Load target image and detect markers
        self.target_image = cv2.imread(target_image_path)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self.parameters = cv2.aruco.DetectorParameters()

        self.target_corners, self.target_ids = self.detect_markers(self.target_image)
        self.target_positions = self.get_marker_positions(self.target_corners)

    def detect_markers(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
        return corners, ids

    def get_marker_positions(self, corners):
        positions = {}
        if corners is not None:
            for i, corner in enumerate(corners):
                c = corner[0]
                top_left = c[0]
                top_right = c[1]
                positions[i] = {'top_left': top_left, 'top_right': top_right}
        return positions

    def calculate_movement(self, live_positions):
        commands = []

        for id in self.target_positions:
            if id in live_positions:
                target_top_left = self.target_positions[id]['top_left']
                target_top_right = self.target_positions[id]['top_right']

                live_top_left = live_positions[id]['top_left']
                live_top_right = live_positions[id]['top_right']

                # Calculate movements based on the difference between target and live positions
                if live_top_left[0] < target_top_left[0] - 20:
                    commands.append("move right")
                elif live_top_left[0] > target_top_left[0] + 20:
                    commands.append("move left")

                if live_top_left[1] < target_top_left[1] - 20:
                    commands.append("move down")
                elif live_top_left[1] > target_top_left[1] + 20:
                    commands.append("move up")

                # Calculate rotation
                target_vector = target_top_right - target_top_left
                live_vector = live_top_right - live_top_left

                target_angle = math.atan2(target_vector[1], target_vector[0])
                live_angle = math.atan2(live_vector[1], live_vector[0])

                angle_diff = math.degrees(live_angle - target_angle)
                if angle_diff > 10:
                    commands.append("turn right")
                elif angle_diff < -10:
                    commands.append("turn left")

        return commands

    def track(self):
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            live_corners, live_ids = self.detect_markers(frame)
            live_positions = self.get_marker_positions(live_corners)

            commands = self.calculate_movement(live_positions)

            # Draw detected markers and their IDs
            if live_ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, live_corners, live_ids)

            # Display movement commands on the frame
            for command in commands:
                cv2.putText(frame, command, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    target_image_path = 'frame_10.jpg'  # Path to the target frame image
    tracker = ArucoTracker(target_image_path)
    tracker.track()
