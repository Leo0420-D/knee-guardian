# rehab_evaluators/Squats.py

import numpy as np
import mediapipe as mp

PoseLandmark = mp.solutions.pose.PoseLandmark


class SquatEvaluator:
    def __init__(self,
                 hip_knee_angle_threshold_down=160,
                 hip_knee_angle_threshold_up=100,
                 feedback_visibility_threshold=0.7
                 ):
        self.hip_knee_angle_threshold_down = hip_knee_angle_threshold_down
        self.hip_knee_angle_threshold_up = hip_knee_angle_threshold_up
        self.feedback_visibility_threshold = feedback_visibility_threshold
        self.reset()

    def _calculate_angle(self, p1, p2, p3):
        p1 = np.array([p1.x, p1.y])
        p2 = np.array([p2.x, p2.y])
        p3 = np.array([p3.x, p3.y])
        v1 = p1 - p2
        v2 = p3 - p2
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 180.0
        angle_rad = np.arccos(np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0))
        return np.degrees(angle_rad)

    def _are_landmarks_visible(self, landmarks, required_indices):
        for index in required_indices:
            if landmarks[index.value].visibility < self.feedback_visibility_threshold:
                return False
        return True

    def evaluate(self, landmarks, current_time):
        if landmarks is None:
            self.feedback_message = "未检测到人体"
            metrics = {"状态": "未知", "计数": self.squat_counter}
            return self.feedback_message, metrics

        required_landmarks = [
            PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER,
            PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP,
            PoseLandmark.LEFT_KNEE, PoseLandmark.RIGHT_KNEE,
            PoseLandmark.LEFT_ANKLE, PoseLandmark.RIGHT_ANKLE
        ]

        if not self._are_landmarks_visible(landmarks, required_landmarks):
            self.feedback_message = "关键点不清晰，请正对摄像头"
            metrics = {"状态": "检测中", "计数": self.squat_counter}
            return self.feedback_message, metrics

        left_hip = landmarks[PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[PoseLandmark.RIGHT_HIP.value]
        left_knee = landmarks[PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[PoseLandmark.RIGHT_KNEE.value]
        left_ankle = landmarks[PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[PoseLandmark.RIGHT_ANKLE.value]

        angle_left_knee = self._calculate_angle(left_hip, left_knee, left_ankle)
        angle_right_knee = self._calculate_angle(right_hip, right_knee, right_ankle)
        avg_knee_angle = (angle_left_knee + angle_right_knee) / 2

        avg_hip_y = (left_hip.y + right_hip.y) / 2
        avg_knee_y = (left_knee.y + right_knee.y) / 2

        self.feedback_message = ""

        if self.squat_stage == "up":
            if avg_knee_angle < 160:
                self.squat_stage = "down"
                self.feedback_message = "下蹲中..."
                self.max_depth_achieved_this_squat = False
        elif self.squat_stage == "down":
            is_deep_enough_hip = avg_hip_y >= avg_knee_y
            is_deep_enough_angle = avg_knee_angle < self.hip_knee_angle_threshold_up

            if is_deep_enough_hip and is_deep_enough_angle:
                self.feedback_message = "保持住！深度达标！"
                self.max_depth_achieved_this_squat = True
            else:
                self.feedback_message = "继续下蹲..."

            if avg_knee_angle > self.hip_knee_angle_threshold_up + 10:
                if self.max_depth_achieved_this_squat:
                    self.squat_counter += 1
                    self.feedback_message = f"很好！第 {self.squat_counter} 个完成！"
                else:
                    self.feedback_message = "起来了，但深度不足！"
                self.squat_stage = "up"
                self.max_depth_achieved_this_squat = False

        if not self.feedback_message:
            self.feedback_message = "请准备开始下一个深蹲" if self.squat_stage == "up" else "下蹲中..."

        metrics = {
            "状态": "站立" if self.squat_stage == "up" else "下蹲",
            "计数": self.squat_counter
        }
        return self.feedback_message, metrics

    def reset(self):
        self.squat_stage = "up"
        self.squat_counter = 0
        self.feedback_message = "请开始深蹲"
        self.max_depth_achieved_this_squat = False
