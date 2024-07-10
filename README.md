# TelloAI V.0.1 - Indoor Autonomous Drone Competition

## Created By:
* :space_invader: [Shaked Levi](https://github.com/20shaked20)
* :octocat: [Dana Zorohov](https://github.com/danaZo)
* :trollface: [Yuval Bubnovsky](https://github.com/YuvalBubnovsky)
</br></br>

## Overview
Welcome to the TelloAI V.0.1 repository for the Indoor Autonomous Drone Competition.
This project focuses on two main objectives:

1. ```ArucoDetector.py```: Detecting ArUco markers in a video, identifying them in each frame, and extracting relevant information about each detected marker. The processed video is saved with the detected markers highlighted.
2. ```ArucoTracker.py```: Tracking and aligning a live video feed with a reference image containing ArUco markers. This script calculates movement commands based on the difference between the live feed and the reference image to help align the camera with the target position.
</br></br>

## Features
- **Real-time ArUco Marker Detection:** The program processes each video frame in real-time to detect ArUco markers. 
- **2D and 3D Marker Information Extraction:** For each detected marker, the program extracts its ID, 2D corner coordinates, and 3D information (distance to the camera, yaw, pitch, roll).
- **Output Formats:** The results are saved in a CSV file and the processed video is saved with the detected markers highlighted in green rectangles.
- **Live Feed Tracking: ** The program tracks a live video feed and provides movement commands to align with a reference image containing ArUco markers.
- **Movement Calculation:** Calculates and displays movement commands, axis differences, and rotation differences between the live feed and the reference image.
</br></br>

## Installation
Clone the repository:

```
git clone https://github.com/20shaked20/ArucoCodeDetection.git
```

Install the required packages:

```
pip install -r requirements.txt
```


## How to Run
### ArucoDetector.py
If you wish to add your own video file follow the instructions, if not you can skip to the "Run the Detection Script".</br>
- **Place Your Video File:** Ensure your video file (e.g., new_video.mp4) is in the same directory as the script or note its relative path.
- **Update the Video Source in the Script:** Open ```ArucoDetector.py``` and locate the section where the video source is defined. Update the ```video_path```, ```output_csv_path``` and ```output_video_path``` variables to point to your new video file and desired output locations:
```
if __name__ == "__main__":
    ...
    video_path = 'new_video.mp4'  # Path to your new video file
    output_csv_path = 'output.csv'  # Path to save the output CSV file
    output_video_path = 'output_video.avi'  # Path to save the output video file
    ...
```
**Run the Detection Script:** Execute the script to process the new video file:
```
python ArucoDetector.py
```

## Parameters:

- ```video_path```: Path to the input video file. Default is ```challengeB.mp4```.</br>
- ```output_csv_path```: Path to save the CSV file. Default is ```output.csv```.</br>
- ```output_video_path```: h to save the output video file with detected markers. Default is ```output_video.avi```. </br></br>

### ArucoTracker.py
- **Place Your Target Image File:** Ensure your target image file (e.g., target_frame_1.jpg) is in the same directory as the script or note its relative path.

- **Update the Target Image Source in the Script:** Open ```ArucoTracker.py``` and locate the section where the target image source is defined. Update the ```target_image_path``` variable to point to your new target image file:
```
if __name__ == "__main__":
    target_image_path = 'target_frame_1.jpg'  # Path to the target frame image
    tracker = ArucoTracker(target_image_path)
    tracker.track()
```
- **Run the Tracking Script:** Execute the script to start tracking:
```
python ArucoTracker.py
```

## Code Explanation
```ArucoDetector.py```</br>
This script contains the main classes and functions for detecting ArUco markers in a video.</br>

**Classes and Methods**
```ArucoDetection```: The main class that handles ArUco detection.
- ```__init__(self, aruco_dict)```: Initializes the class with the specified ArUco dictionary and relevant parameters.
- ```set_image_to_process(self, img)```: Sets the image to be processed.
- ```detect_aruco(self)```: Detects ArUco markers in the image in a separate thread.
- ```draw_detection(self, image)```: Draws detected markers on the image.
- ```process_video(self, video_source, output_path)```: Processes the video, detects markers, and saves the output.
</br>

```ArucoTracker.py```</br>
This script contains the main classes and functions for tracking movement based on ArUco markers.

**Classes and Methods**
```ArucoTracker```: The main class that handles ArUco tracking.
- ```__init__(self, target_image_path, marker_size=0.05)```: Initializes the class with the specified target image and relevant parameters.
- ```detect_markers(self, image)```: Detects ArUco markers in the image.
- ```calculate_movement(self, live_rvecs, live_tvecs)```: Calculates movement commands based on the difference between target and live marker positions.
- ```track(self)```: Tracks the live video feed and issues movement commands.
</br></br>


## Output
The output consists of two files:

- **CSV File:** Contains the frame ID, ArUco ID, 2D coordinates of the marker corners, and 3D information (distance, yaw).
- **Output Video:** The input video with detected ArUco markers highlighted in green rectangles.
</br></br>

## Requirements
- Python 3.7+
- OpenCV 4.5+
- Imutils
- Numpy
- open-cv-contrib
</br></br>

## References
- [Ryze Tello Specifications](https://www.ryzerobotics.com/tello/specs)
- [OpenCV ArUco Module](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)
</br></br>

