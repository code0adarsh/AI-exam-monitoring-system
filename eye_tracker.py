import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from dataclasses import dataclass
from typing import Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@dataclass
class EyeMetrics:
    ear: float
    gaze_direction: str
    major_eye_movements: int
    face_rotation_detected: bool

class EyeTracker:
    def __init__(self, 
                 blink_threshold: float = 0.2,
                 major_eye_movement_threshold: float = 0.3,
                 gaze_left_threshold: float = 0.8,
                 gaze_right_threshold: float = 1.2,
                 face_rotation_threshold: float = 30.0):  # Initial high threshold
        """Initialize the eye tracker with configurable parameters."""
        self.blink_threshold = blink_threshold
        self.major_eye_movement_threshold = major_eye_movement_threshold
        self.gaze_left_threshold = gaze_left_threshold
        self.gaze_right_threshold = gaze_right_threshold
        self.face_rotation_threshold = face_rotation_threshold
        
        # State variables
        self.major_eye_movements_count = 0
        self.last_gaze_direction = None
        
        # Initialize face detection models
        try:
            self.face_detector = dlib.get_frontal_face_detector()
            self.landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        except RuntimeError as e:
            logging.error(f"Failed to load face detection models: {e}")
            raise

    @staticmethod
    def calculate_ear(eye: np.ndarray) -> float:
        """Calculate the Eye Aspect Ratio (EAR)."""
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C) if C > 0 else 0

    def detect_gaze(self, eye: np.ndarray, frame_gray: np.ndarray) -> str:
        """Detect gaze direction."""
        x, y, w, h = cv2.boundingRect(eye)
        if w == 0 or h == 0:
            return "Center"
        
        roi = frame_gray[y:y + h, x:x + w]
        blurred = cv2.GaussianBlur(roi, (7, 7), 0)
        _, threshold = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY)
        
        h, w = threshold.shape
        left_white = cv2.countNonZero(threshold[:, :w // 2])
        right_white = cv2.countNonZero(threshold[:, w // 2:])
        
        gaze_ratio = left_white / (right_white + 0.1)
        if gaze_ratio < self.gaze_left_threshold:
            return "Right"
        elif gaze_ratio > self.gaze_right_threshold:
            return "Left"
        return "Center"

    def detect_face_rotation(self, landmarks) -> bool:
        """Detect face rotation based on 3D pose."""
        image_points = np.array([
            (landmarks.part(30).x, landmarks.part(30).y),  # Nose tip
            (landmarks.part(8).x, landmarks.part(8).y),    # Chin
            (landmarks.part(36).x, landmarks.part(36).y),  # Left eye left corner
            (landmarks.part(45).x, landmarks.part(45).y),  # Right eye right corner
            (landmarks.part(48).x, landmarks.part(48).y),  # Left mouth corner
            (landmarks.part(54).x, landmarks.part(54).y)   # Right mouth corner
        ], dtype="double")

        # 3D model points of a generic head
        model_points = np.array([
            (0.0, 0.0, 0.0),         # Nose tip
            (0.0, -330.0, -65.0),    # Chin
            (-225.0, 170.0, -135.0), # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corner
            (-150.0, -150.0, -125.0),# Left mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner
        ])

        # Camera matrix (assuming typical focal length and center)
        size = (640, 480)  # Typical webcam resolution
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

        # Solve PnP to find the pose
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return False

        # Convert rotation vector to Euler angles
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        euler_angles = cv2.decomposeProjectionMatrix(
            np.hstack((rotation_matrix, translation_vector)))[6]

        yaw = abs(euler_angles[1])  # Yaw angle (left-right rotation)

        # Detect if yaw angle exceeds threshold
        return yaw > self.face_rotation_threshold

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Process a single frame and return the annotated frame."""
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(frame_gray)
        
        for face in faces:
            landmarks = self.landmark_predictor(frame_gray, face)
            
            left_eye = self._get_eye_coordinates(landmarks, 36, 42)
            right_eye = self._get_eye_coordinates(landmarks, 42, 48)
            
            left_ear = self.calculate_ear(left_eye)
            right_ear = self.calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2
            
            if avg_ear < self.major_eye_movement_threshold:
                self.major_eye_movements_count += 1

            face_rotation_detected = self.detect_face_rotation(landmarks)

            cv2.putText(frame, f"Major Eye Movements: {self.major_eye_movements_count}", 
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Face Rotation Detected: {face_rotation_detected}", 
                        (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame, False

    @staticmethod
    def _get_eye_coordinates(landmarks, start: int, end: int) -> np.ndarray:
        """Extract eye coordinates."""
        return np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(start, end)])

def main():
    tracker = EyeTracker()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame, _ = tracker.process_frame(frame)
        cv2.imshow("Eye & Face Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
