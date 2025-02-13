import cv2
import dlib
import numpy as np
import mediapipe as mp
from transformers import DetrFeatureExtractor, DetrForObjectDetection
import torch
from PIL import Image
from dataclasses import dataclass
from typing import Tuple, List
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@dataclass
class ExamMetrics:
    face_rotation: bool
    neck_movement: bool
    detected_objects: List[Tuple[str, float, str, List[float]]]  # (object_name, score, legal_status, bbox)

class ExamMonitor:
    def __init__(self, 
                 face_rotation_threshold: float = 30.0,
                 neck_movement_threshold: float = 15.0,
                 allowed_items: dict = None):
        """
        Initialize the exam monitor.
        - face_rotation_threshold: Yaw angle in degrees above which head rotation is flagged.
        - neck_movement_threshold: Normalized vertical displacement threshold for neck movement.
        - allowed_items: Mapping of object names (lowercase) to "legal" or "illegal" strings.
        """
        # Default allowed items mapping; customize as needed.
        if allowed_items is None:
            allowed_items = {"laptop": "legal", "phone": "illegal", "tablet": "illegal"}
        self.allowed_items = allowed_items
        self.face_rotation_threshold = face_rotation_threshold
        self.neck_movement_threshold = neck_movement_threshold

        # Initialize dlib face detector and landmark predictor.
        try:
            self.face_detector = dlib.get_frontal_face_detector()
            self.landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        except RuntimeError as e:
            logging.error(f"Failed to load dlib models: {e}")
            raise

        # Initialize Mediapipe Pose for neck movement detection.
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

        # Initialize HuggingFace DETR model for object detection.
        logging.info("Loading HuggingFace DETR model...")
        self.feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
        self.object_detector = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        logging.info("Models loaded successfully.")

    def detect_face_rotation(self, landmarks) -> bool:
        """Detect head rotation using a 3D pose estimation from face landmarks."""
        # Define 2D image points from selected facial landmarks.
        image_points = np.array([
            (landmarks.part(30).x, landmarks.part(30).y),  # Nose tip
            (landmarks.part(8).x, landmarks.part(8).y),    # Chin
            (landmarks.part(36).x, landmarks.part(36).y),  # Left eye left corner
            (landmarks.part(45).x, landmarks.part(45).y),  # Right eye right corner
            (landmarks.part(48).x, landmarks.part(48).y),  # Left mouth corner
            (landmarks.part(54).x, landmarks.part(54).y)   # Right mouth corner
        ], dtype="double")

        # 3D model points of a generic head.
        model_points = np.array([
            (0.0, 0.0, 0.0),         # Nose tip
            (0.0, -330.0, -65.0),    # Chin
            (-225.0, 170.0, -135.0), # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corner
            (-150.0, -150.0, -125.0),# Left mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner
        ])

        # Assume a typical webcam resolution.
        size = (640, 480)
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion.
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return False

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        euler_angles = cv2.decomposeProjectionMatrix(
            np.hstack((rotation_matrix, translation_vector))
        )[6]
        yaw = abs(euler_angles[1])
        return yaw > self.face_rotation_threshold

    def detect_neck_movement(self, frame: np.ndarray) -> bool:
        """Detect neck movement using Mediapipe Pose landmarks."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        if not results.pose_landmarks:
            return False

        # Use nose and shoulder landmarks.
        nose = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        left_shoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Calculate the midpoint of the shoulders.
        mid_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2

        # Compute vertical displacement between nose and mid-shoulder.
        vertical_displacement = abs(nose.y - mid_shoulder_y)
        # Compare with a threshold (normalized coordinates: threshold adjusted as needed).
        return vertical_displacement > (self.neck_movement_threshold / 100.0)

    def detect_objects(self, frame: np.ndarray) -> List[Tuple[str, float, str, List[float]]]:
        """Detect objects in the frame using DETR and classify them as legal/illegal."""
        # Convert frame to a PIL Image.
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        outputs = self.object_detector(**inputs)

        # Post-process predictions.
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.feature_extractor.post_process(outputs, target_sizes=target_sizes)[0]
        detected_objects = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if score < 0.9:  # Filter low-confidence predictions.
                continue
            object_name = self.object_detector.config.id2label[label.item()]
            # Determine status based on allowed_items mapping (default to legal if not specified).
            status = self.allowed_items.get(object_name.lower(), "legal")
            detected_objects.append((object_name, score.item(), status, box.tolist()))
        return detected_objects

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, ExamMetrics]:
        """Process a single frame: detect face/neck movement and objects, annotate frame."""
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(frame_gray)
        face_rotation_flag = False

        # Process each detected face.
        for face in faces:
            landmarks = self.landmark_predictor(frame_gray, face)
            if self.detect_face_rotation(landmarks):
                face_rotation_flag = True
            # Draw face bounding box.
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)

        # Detect neck movement.
        neck_movement_flag = self.detect_neck_movement(frame)

        # Detect objects and annotate them.
        objects = self.detect_objects(frame)
        for obj_name, score, status, box in objects:
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0) if status == "legal" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label_text = f"{obj_name} ({status})"
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Annotate face and neck movement results.
        cv2.putText(frame, f"Face Rotation: {face_rotation_flag}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Neck Movement: {neck_movement_flag}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        metrics = ExamMetrics(
            face_rotation=face_rotation_flag,
            neck_movement=neck_movement_flag,
            detected_objects=objects
        )
        return frame, metrics

def main():
    monitor = ExamMonitor()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logging.error("Could not open video capture.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning("Failed to grab frame.")
            break

        frame, metrics = monitor.process_frame(frame)
        cv2.imshow("Exam Monitor", frame)

        # Press 'q' to exit.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
