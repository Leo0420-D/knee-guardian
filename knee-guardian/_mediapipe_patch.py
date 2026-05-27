"""
MediaPipe compatibility patch: provides the old mp.solutions API
using the new mp.tasks API (MediaPipe 0.10.x).
"""
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import PoseLandmark, PoseLandmarksConnections
import os
import urllib.request

MODEL_URL = "https://storage.googleapis.com/mediapipe-assets/pose_landmarker.task"


def _get_model_path():
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    model_path = os.path.join(model_dir, "pose_landmarker.task")
    if not os.path.exists(model_path):
        os.makedirs(model_dir, exist_ok=True)
        print(f"Downloading PoseLandmarker model to {model_path}...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print("Model download complete.")
    return model_path


class _LandmarkListWrapper:
    """Make new API list[NormalizedLandmark] behave like old NormalizedLandmarkList."""
    def __init__(self, landmarks):
        self.landmark = list(landmarks)

    def __iter__(self):
        return iter(self.landmark)

    def __len__(self):
        return len(self.landmark)

    def __getitem__(self, i):
        return self.landmark[i]

    def __bool__(self):
        return True


class _ResultWrapper:
    def __init__(self, result):
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            self.pose_landmarks = _LandmarkListWrapper(result.pose_landmarks[0])
            self.pose_world_landmarks = None
            if result.pose_world_landmarks and len(result.pose_world_landmarks) > 0:
                self.pose_world_landmarks = _LandmarkListWrapper(result.pose_world_landmarks[0])
        else:
            self.pose_landmarks = None
            self.pose_world_landmarks = None


class Pose:
    def __init__(self, static_image_mode=False, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5, model_complexity=1,
                 smooth_landmarks=True, enable_segmentation=False,
                 smooth_segmentation=True):
        model_path = _get_model_path()
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    def process(self, image):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        result = self._landmarker.detect(mp_image)
        return _ResultWrapper(result)

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class _PoseModule:
    Pose = Pose
    PoseLandmark = PoseLandmark
    POSE_CONNECTIONS = PoseLandmarksConnections


def _draw_landmarks(image, landmark_list, connections=None,
                    landmark_drawing_spec=None, connection_drawing_spec=None):
    if landmark_list is not None and hasattr(landmark_list, 'landmark'):
        landmark_list = landmark_list.landmark
    return mp_vision.drawing_utils.draw_landmarks(
        image, landmark_list, connections,
        landmark_drawing_spec=landmark_drawing_spec,
        connection_drawing_spec=connection_drawing_spec)


class _DrawingUtils:
    draw_landmarks = staticmethod(_draw_landmarks)
    DrawingSpec = mp_vision.drawing_utils.DrawingSpec


class _Solutions:
    pose = _PoseModule()
    drawing_utils = _DrawingUtils()


mp.solutions = _Solutions()
