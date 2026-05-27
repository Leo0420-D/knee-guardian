# rehab_evaluators/Neckrotation.py

import numpy as np
import mediapipe as mp
import time

PoseLandmark = mp.solutions.pose.PoseLandmark


class NeckRotationEvaluator:
    def __init__(self,
                 rotation_angle_threshold=40,
                 hold_time_threshold=3,
                 feedback_visibility_threshold=0.7
                 ):
        self.rotation_angle_threshold = rotation_angle_threshold
        self.hold_time_threshold = hold_time_threshold
        self.feedback_visibility_threshold = feedback_visibility_threshold
        self.reset()

    def _calculate_angle(self, p1, p2, p3):
        # This 2D angle calculation for neck rotation is a simplification.
        # It's based on the horizontal positions of nose vs shoulders.
        nose_x, nose_y = p2.x, p2.y
        left_shoulder_x, left_shoulder_y = p1.x, p1.y
        right_shoulder_x, right_shoulder_y = p3.x, p3.y

        shoulder_center_x = (left_shoulder_x + right_shoulder_x) / 2
        shoulder_width = abs(left_shoulder_x - right_shoulder_x)

        if shoulder_width == 0: return 0

        # rotation_ratio: -1 (full left), 0 (center), 1 (full right)
        rotation_ratio = (nose_x - shoulder_center_x) / (shoulder_width / 2)
        # Convert to an angle-like value, e.g., 90 degrees is max rotation
        rotation_angle = abs(rotation_ratio * 90)

        return rotation_angle, "left" if rotation_ratio < 0 else "right"

    def _are_landmarks_visible(self, landmarks, required_indices):
        for index in required_indices:
            if landmarks[index.value].visibility < self.feedback_visibility_threshold:
                return False
        return True

    def evaluate(self, landmarks, current_time):
        if landmarks is None:
            metrics = {"状态": "未知", "计数": self.rotation_counter}
            return "未检测到人体", metrics

        required_landmarks = [
            PoseLandmark.NOSE,
            PoseLandmark.LEFT_SHOULDER,
            PoseLandmark.RIGHT_SHOULDER
        ]

        if not self._are_landmarks_visible(landmarks, required_landmarks):
            metrics = {"状态": "检测中", "计数": self.rotation_counter}
            return "关键点不清晰，请正对摄像头", metrics

        nose = landmarks[PoseLandmark.NOSE.value]
        left_shoulder = landmarks[PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[PoseLandmark.RIGHT_SHOULDER.value]

        angle, direction = self._calculate_angle(left_shoulder, nose, right_shoulder)

        if self.rotation_stage == "neutral":
            if angle > self.rotation_angle_threshold:
                self.rotation_stage = direction
                self.hold_start_time = current_time
                self.feedback_message = f"向{self.rotation_stage}转动，保持住"

        elif self.rotation_stage in ["left", "right"]:
            if direction != self.rotation_stage or angle < self.rotation_angle_threshold - 10:
                self.feedback_message = "动作中断，请回正"
                self.rotation_stage = "neutral"
                self.hold_start_time = None
            else:
                elapsed = current_time - self.hold_start_time
                if elapsed >= self.hold_time_threshold:
                    self.rotation_counter += 1
                    self.feedback_message = f"很好，完成第 {self.rotation_counter} 次！请回正"
                    self.rotation_stage = "completed"
                else:
                    self.feedback_message = f"保持... {self.hold_time_threshold - elapsed:.1f} 秒"

        elif self.rotation_stage == "completed":
            if angle < 15:  # Threshold for returning to neutral
                self.rotation_stage = "neutral"
                self.feedback_message = "请向另一侧转动或休息"

        metrics = {
            "状态": self.rotation_stage,
            "计数": self.rotation_counter
        }

        if self.rotation_stage == "neutral" and not self.feedback_message:
            self.feedback_message = "请缓慢向左或向右转头"

        return self.feedback_message, metrics

    def reset(self):
        self.rotation_stage = "neutral"
        self.rotation_counter = 0
        self.feedback_message = "请开始颈部旋转"
        self.hold_start_time = None
