# rehab_evaluators/sitandreach.py

import numpy as np
import mediapipe as mp
import time

PoseLandmark = mp.solutions.pose.PoseLandmark


class SitAndReachEvaluator:
    def __init__(self,
                 hip_angle_threshold_down=120,
                 hip_angle_threshold_up=60,
                 feedback_visibility_threshold=0.7
                 ):
        self.hip_angle_threshold_down = hip_angle_threshold_down
        self.hip_angle_threshold_up = hip_angle_threshold_up
        self.feedback_visibility_threshold = feedback_visibility_threshold
        self.reset()

    def _calculate_angle(self, p1, p2, p3):
        p1, p2, p3 = np.array([p1.x, p1.y]), np.array([p2.x, p2.y]), np.array([p3.x, p3.y])
        v1, v2 = p1 - p2, p3 - p2
        dot, norm = np.dot(v1, v2), np.linalg.norm(v1) * np.linalg.norm(v2)
        return np.degrees(np.arccos(np.clip(dot / norm, -1.0, 1.0))) if norm != 0 else 180.0

    def _are_landmarks_visible(self, landmarks, required_indices):
        return all(landmarks[i.value].visibility >= self.feedback_visibility_threshold for i in required_indices)

    def evaluate(self, landmarks, current_time):
        if landmarks is None:
            self.feedback_message = "未检测到人体"
        else:
            required = [PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER, PoseLandmark.LEFT_HIP,
                        PoseLandmark.RIGHT_HIP, PoseLandmark.LEFT_KNEE, PoseLandmark.RIGHT_KNEE]
            if not self._are_landmarks_visible(landmarks, required):
                self.feedback_message = "关键点不清晰，请侧对摄像头"
            else:
                l_shoulder, r_shoulder = landmarks[PoseLandmark.LEFT_SHOULDER.value], landmarks[
                    PoseLandmark.RIGHT_SHOULDER.value]
                l_hip, r_hip = landmarks[PoseLandmark.LEFT_HIP.value], landmarks[PoseLandmark.RIGHT_HIP.value]
                l_knee, r_knee = landmarks[PoseLandmark.LEFT_KNEE.value], landmarks[PoseLandmark.RIGHT_KNEE.value]

                avg_hip_angle = (self._calculate_angle(l_shoulder, l_hip, l_knee) + self._calculate_angle(r_shoulder,
                                                                                                          r_hip,
                                                                                                          r_knee)) / 2

                if self.stage == "up":
                    if avg_hip_angle < self.hip_angle_threshold_down:
                        self.stage = "down"
                        self.feedback_message = "前屈中..."
                        self.max_depth_achieved = False
                elif self.stage == "down":
                    if avg_hip_angle < self.hip_angle_threshold_up:
                        self.feedback_message = "达到最大幅度！"
                        self.max_depth_achieved = True

                    if avg_hip_angle > self.hip_angle_threshold_down - 10:
                        if self.max_depth_achieved:
                            self.counter += 1
                            self.feedback_message = f"很好！完成第 {self.counter} 次！"
                        else:
                            self.feedback_message = "幅度不足，请再次尝试！"
                        self.stage = "up"

        if not self.feedback_message and self.stage == "up":
            self.feedback_message = "请准备开始下一个坐位体前屈"
        elif not self.feedback_message and self.stage == "down":
            self.feedback_message = "努力向前伸..."

        metrics = {"状态": "坐直" if self.stage == "up" else "前屈", "计数": self.counter}
        return self.feedback_message, metrics

    def reset(self):
        self.stage = "up"
        self.counter = 0
        self.feedback_message = "请坐直并开始"
        self.max_depth_achieved = False
