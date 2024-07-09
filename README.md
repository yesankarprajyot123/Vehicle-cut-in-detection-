# Vehicle Cut-In Detection

## Overview

This project focuses on detecting vehicle cut-ins using a combination of computer vision techniques and machine learning models. The solution employs the YOLO (You Only Look Once) model for real-time object detection and OpenCV for image processing tasks. The primary objective is to accurately detect and warn about vehicles cutting into the lane, enhancing the safety of autonomous driving systems.

## Introduction

Vehicle cut-in detection is a critical aspect of advanced driver-assistance systems (ADAS) and autonomous driving. The ability to detect vehicles cutting into the lane can significantly reduce the risk of collisions and improve overall road safety. This project leverages YOLO for object detection and calculates various parameters to determine potential cut-in scenarios.

## Requirements

- Python 3.7 or higher
- OpenCV
- ultralytics
- numpy

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yesankarprajyot123/vehicle-cut-in-detection.git
   cd vehicle-cut-in-detection
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Ensure you have a YOLO model file (`best.pt`) located in the `model/` directory.
2. Prepare your image dataset and place it in a directory, e.g., `./images`.
3. Run the cut-in detection script:
   ```bash
   python cut_in_detection.py <image_directory_name>
   ```
   Example:
   ```bash
   python cut_in_detection.py ./images
   ```

## Methodology

### Frame Dimensions and Lane Markings
The script initializes frame dimensions and lane markings to set up the reference points for detecting cut-ins.

### Distance Calculation
The distance to the detected vehicles is calculated using the height of the bounding box and the known dimensions of typical vehicles.

### Cut-In Detection Logic
1. **Bounding Box Adjustment**: Adjust the bounding box coordinates based on the frame size.
2. **Distance Tracking**: Track the distance of each detected vehicle over time.
3. **Velocity and Angle Calculation**: Calculate the velocity and angle of the detected vehicles to determine potential cut-in events.
4. **Warning Generation**: Generate warnings for vehicles that exhibit cut-in behavior based on calculated Time-To-Collision (TTC) and angular deviation.

## Results

The system displays a warning for detected cut-ins with visual indicators on the processed frames. The results are shown in real-time with FPS (Frames Per Second) information to monitor performance.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The authors of the YOLO model for providing an efficient object detection framework.
- The OpenCV community for their extensive resources and support.
