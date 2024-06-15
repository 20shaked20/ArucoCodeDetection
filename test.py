import threading
import cv2
import numpy as np
import imutils

class ArucoDetection:

    def __init__(self, aruco_dict, camera_matrix, dist_coeffs, marker_length):
        self.img = None
        self.ARUCO_DICT = aruco_dict
        self.camera_matrix = np.array(camera_matrix, dtype=np.float32)
        self.dist_coeffs = np.array(dist_coeffs, dtype=np.float32)
        self.marker_length = marker_length  # Length of the ArUco marker in meters
        self.corners = []
        self.ids = []
        self.rvecs = []
        self.tvecs = []

        self.lock = threading.Lock()
        self.stop_event = threading.Event()

    def set_image_to_process(self, img):
        """Set the image to process."""
        with self.lock:
            self.img = img.copy()

    def preprocess_frame(self, frame):
        """Preprocess the frame to improve detection accuracy."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        return blur

    def detect_aruco(self):
        """Detect ArUco markers in the image."""
        while not self.stop_event.is_set():
            with self.lock:
                if self.img is not None:
                    img = self.img.copy()
                else:
                    continue

            image = imutils.resize(img, width=1080)  # Process at higher resolution
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

    def draw_detection(self, image):
        """Draw detections on the image."""
        with self.lock:
            if self.ids is not None and self.corners is not None:
                for corner, id_, rvec, tvec in zip(self.corners, self.ids, self.rvecs, self.tvecs):
                    cv2.aruco.drawDetectedMarkers(image, [corner], id_)
                    cv2.drawFrameAxes(image, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_length * 0.5)
        return image

    def process_video(self, video_source, output_path):
        """Process video frames to detect ArUco codes and export the result."""
        cap = cv2.VideoCapture(video_source)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for the output video
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        threading.Thread(target=self.detect_aruco, daemon=True).start()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            self.set_image_to_process(frame)
            frame = self.draw_detection(frame)
            out.write(frame)

            cv2.imshow('Aruco Detection', frame)
            if cv2.waitKey(int(300 / fps)) & 0xFF == ord('q'):
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

    camera_matrix = np.array([[921.170702, 0.000000, 459.904354], [0.000000, 919.018377, 351.238301], [0.000000, 0.000000, 1.000000]])
    dist_coeffs = np.array([-0.033458, 0.105152, 0.001256, -0.006647, 0.000000])
    marker_length = 0.05  # Marker length in meters (example value)

    detection = ArucoDetection(aruco_dict, camera_matrix, dist_coeffs, marker_length)
    video_source = "challengeB.mp4"  
    output_path = 'processed_output_video.mp4'  
    detection.process_video(video_source, output_path)
