# AI Exam Proctor using Camera Detection

An advanced exam monitoring system that integrates face and neck movement detection with object recognition (e.g., detecting phones, laptops, etc.) using dlib, Mediapipe, and Hugging Face’s DETR model. This project is designed to help ensure exam integrity by flagging unauthorized objects and suspicious head/neck movements.

## Features

- **Face & Head Pose Detection:** Uses dlib’s 68-point facial landmark model for detecting head rotation.
- **Neck Movement Detection:** Utilizes Mediapipe’s pose estimation to monitor neck movement.
- **Object Detection & Classification:** Integrates Hugging Face’s DETR model to detect and categorize objects (e.g., phone, laptop) as legal or illegal based on a configurable mapping.
- **Real-Time Video Processing:** Processes video from a webcam and displays annotated frames.
- **Configurable Thresholds:** Customize detection thresholds for face rotation, neck movement, and object classification.

## Installation

### Prerequisites

Ensure that you have Python 3.8 or higher installed along with pip. This project depends on several Python packages and one external model file.

### Python Dependencies

Install the required Python packages using pip. You can use the provided **requirements.txt** file by running:

```bash
pip install -r requirements.txt
```

## External Model File
The project requires dlib’s 68-point facial landmark predictor file (shape_predictor_68_face_landmarks.dat), which is not bundled with dlib by default. 
Download the compressed model file from dlib's official website. After downloading, extract the file using a tool like 7-Zip or by running the following command in your terminal:

```bash
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
```

## Usage
To run the exam monitoring application, simply execute the main script:

```bash
python main.py
```
