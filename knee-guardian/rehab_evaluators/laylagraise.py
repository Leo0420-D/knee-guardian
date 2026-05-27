# rehab_evaluators/laylegraise.py

import numpy as np
import mediapipe as mp
import time

PoseLandmark = mp.solutions.pose.PoseLandmark


class LayLegRaiseEvaluator:
    def __init__(self,
                 hip_angle_threshold_down=160,
                 hip_angle_threshold_up=120,
                 feedback_visibility_threshold=0.7
                 ):
        self.hip_angle_threshold_down = hip_angle_threshold_down
        self.hip_angle_threshold_up = hip_angle_threshold_up
        self.feedback_visibility_threshold = feedback_visibility_threshold
        self.reset()

    def _calculate_angle(self, p1, p2, p3):
        p1, p2, p3 = np.array([p1.x, p1.y]), np.array([p2.x, p2.y]), np.array([p3.x, p3.y])
        v1, v2 = p1 - p2, p3 - p2
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        return np.degrees(np.arccos(np.clip(dot / norm, -1.0, 1.0))) if norm != 0 else 180.0

    def _are_landmarks_visible(self, landmarks, required_indices):
        return all(landmarks[i.value].visibility >= self.feedback_visibility_threshold for i in required_indices)

    def evaluate(self, landmarks, current_time):
        if landmarks is None or not self._are_landmarks_visible(landmarks,
                                                                [PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_HIP,
                                                                 PoseLandmark.LEFT_KNEE]):
            self.feedback_message = "关键点不清晰，请侧躺并确保身体可见"
        else:
            l_shoulder = landmarks[PoseLandmark.LEFT_SHOULDER.value]
            l_hip = landmarks[PoseLandmark.LEFT_HIP.value]
            l_knee = landmarks[PoseLandmark.LEFT_KNEE.value]

            # For leg raise, we typically care about one leg at a time
            # Assume user is lying on their side, so we use one side's landmarks
            angle_hip = self._calculate_angle(l_shoulder, l_hip, l_knee)

            if self.stage == "down":
                if angle_hip < self.hip_angle_threshold_down:
                    self.stage = "up"
                    self.feedback_message = "抬腿中..."
                    self.max_height_achieved = False
            elif self.stage == "up":
                if angle_hip < self.hip_angle_threshold_up:
                    self.feedback_message = "达到最高点！"
                    self.max_height_achieved = True

                if angle_hip > self.hip_angle_threshold_down - 5:  # Returned to starting position
                    if self.max_height_achieved:
                        self.counter += 1
                        self.feedback_message = f"很好！完成第 {self.counter} 次！"
                    else:
                        self.feedback_message = "幅度不足，请抬得更高！"
                    self.stage = "down"

        if not self.feedback_message:
            self.feedback_message = "请准备" if self.stage == "down" else "抬腿中..."

        metrics = {"状态": "放下" if self.stage == "down" else "抬起", "计数": self.counter}
        return self.feedback_message, metrics

    def reset(self):
        self.stage = "down"
        self.counter = 0
        self.feedback_message = "请开始仰卧抬腿"
        self.max_height_achieved = False
