# rehab_evaluators/shoulderflexion.py

import numpy as np
import mediapipe as mp
import time

PoseLandmark = mp.solutions.pose.PoseLandmark


class ShoulderFlexionEvaluator:
    def __init__(self,
                 angle_threshold_up=160,
                 angle_threshold_down=40,
                 feedback_visibility_threshold=0.7
                 ):
        self.angle_threshold_up = angle_threshold_up
        self.angle_threshold_down = angle_threshold_down
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
        required = [PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP, PoseLandmark.LEFT_SHOULDER,
                    PoseLandmark.RIGHT_SHOULDER, PoseLandmark.LEFT_ELBOW, PoseLandmark.RIGHT_ELBOW]
        if landmarks is None or not self._are_landmarks_visible(landmarks, required):
            self.feedback_message = "关键点不清晰，请正对摄像头"
        else:
            l_hip, r_hip = landmarks[PoseLandmark.LEFT_HIP.value], landmarks[PoseLandmark.RIGHT_HIP.value]
            l_shoulder, r_shoulder = landmarks[PoseLandmark.LEFT_SHOULDER.value], landmarks[
                PoseLandmark.RIGHT_SHOULDER.value]
            l_elbow, r_elbow = landmarks[PoseLandmark.LEFT_ELBOW.value], landmarks[PoseLandmark.RIGHT_ELBOW.value]

            # Using Hip-Shoulder-Elbow angle
            angle_left = self._calculate_angle(l_hip, l_shoulder, l_elbow)
            angle_right = self._calculate_angle(r_hip, r_shoulder, r_elbow)
            avg_shoulder_angle = (angle_left + angle_right) / 2

            if self.movement_stage == "down":
                if avg_shoulder_angle > self.angle_threshold_up:
                    self.movement_stage = "up"
                    self.feedback_message = "上举中...保持住！"
                    self.max_angle_achieved = True
            elif self.movement_stage == "up":
                if avg_shoulder_angle < self.angle_threshold_down:
                    if self.max_angle_achieved:
                        self.movement_counter += 1
                        self.feedback_message = f"很好！第 {self.movement_counter} 次完成！"
                    else:
                        self.feedback_message = "幅度不足！"
                    self.movement_stage = "down"
                    self.max_angle_achieved = False

        if not self.feedback_message:
            self.feedback_message = "请准备开始" if self.movement_stage == "down" else "上举中..."

        metrics = {"状态": "放下" if self.movement_stage == "down" else "上举", "计数": self.movement_counter}
        return self.feedback_message, metrics

    def reset(self):
        self.movement_stage = "down"
        self.movement_counter = 0
        self.feedback_message = "请开始肩关节前屈上举"
        self.max_angle_achieved = False
