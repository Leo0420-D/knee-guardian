# rehab_evaluators/hipbridge.py

import numpy as np
import time
import mediapipe as mp

PoseLandmark = mp.solutions.pose.PoseLandmark


class HipBridgeTimer:
    def __init__(self, hip_angle_threshold=160, visibility_threshold=0.7):
        self.hip_angle_threshold = hip_angle_threshold
        self.visibility_threshold = visibility_threshold
        self.reset()

    def _calculate_angle(self, p1, p2, p3):
        p1, p2, p3 = np.array([p1.x, p1.y]), np.array([p2.x, p2.y]), np.array([p3.x, p3.y])
        v1, v2 = p1 - p2, p3 - p2
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        return np.degrees(np.arccos(np.clip(dot / norm, -1.0, 1.0))) if norm != 0 else 180.0

    def _check_landmarks_visibility(self, landmarks):
        required = [PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER, PoseLandmark.LEFT_HIP,
                    PoseLandmark.RIGHT_HIP, PoseLandmark.LEFT_KNEE, PoseLandmark.RIGHT_KNEE]
        return all(landmarks[landmark.value].visibility >= self.visibility_threshold for landmark in required)

    def evaluate(self, landmarks, current_time):
        if not landmarks or not self._check_landmarks_visibility(landmarks):
            self.feedback_message = "关键点不清晰，请侧对摄像头"
            if self.current_stage == "bridging": self.current_stage = "failed"
        else:
            l_shoulder, r_shoulder = landmarks[PoseLandmark.LEFT_SHOULDER.value], landmarks[
                PoseLandmark.RIGHT_SHOULDER.value]
            l_hip, r_hip = landmarks[PoseLandmark.LEFT_HIP.value], landmarks[PoseLandmark.RIGHT_HIP.value]
            l_knee, r_knee = landmarks[PoseLandmark.LEFT_KNEE.value], landmarks[PoseLandmark.RIGHT_KNEE.value]

            avg_angle = (self._calculate_angle(l_shoulder, l_hip, l_knee) + self._calculate_angle(r_shoulder, r_hip,
                                                                                                  r_knee)) / 2
            is_proper_bridge = avg_angle >= self.hip_angle_threshold

            if self.current_stage == "not_started":
                if is_proper_bridge:
                    self.current_stage = "bridging"
                    self.start_time = current_time
                    self.feedback_message = "姿势正确！开始计时"
                else:
                    self.feedback_message = "请抬高臀部"
            elif self.current_stage == "bridging":
                if not is_proper_bridge:
                    self.current_stage = "failed"
                    self.session_best = max(self.session_best, current_time - self.start_time)
                    self.feedback_message = f"姿势不正确！上次坚持: {current_time - self.start_time:.1f}秒"
                else:
                    self.elapsed_time = current_time - self.start_time
                    self.feedback_message = f"保持中... {self.elapsed_time:.1f}秒"
            elif self.current_stage in ["completed", "failed"]:
                if is_proper_bridge:
                    self.current_stage = "bridging"
                    self.start_time = current_time
                    self.feedback_message = "重新开始计时"
                else:
                    self.feedback_message = f"最佳成绩: {self.session_best:.1f}秒 | 请重新开始"

        metrics = {"状态": self.current_stage, "计时": f"{self.elapsed_time:.1f}s", "最佳": f"{self.session_best:.1f}s"}
        return self.feedback_message, metrics

    def reset(self):
        self.current_stage = "not_started"
        self.start_time = None
        self.elapsed_time = 0
        self.feedback_message = "请躺好并准备开始臀桥"
        self.session_best = 0
