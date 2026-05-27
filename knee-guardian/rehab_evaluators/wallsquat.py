# rehab_evaluators/wallsquat.py

import numpy as np
import mediapipe as mp
import time

PoseLandmark = mp.solutions.pose.PoseLandmark


class WallSquatEvaluator:
    def __init__(self,
                 knee_angle_threshold=100,  # Target angle is around 90, so threshold is a bit wider
                 hold_time_min=30,
                 feedback_visibility_threshold=0.7
                 ):
        self.knee_angle_threshold = knee_angle_threshold
        self.hold_time_min = hold_time_min
        self.feedback_visibility_threshold = feedback_visibility_threshold
        self.reset()

    def _calculate_angle(self, p1, p2, p3):
        p1, p2, p3 = np.array([p1.x, p1.y]), np.array([p2.x, p2.y]), np.array([p3.x, p3.y])
        v1, v2 = p1 - p2, p3 - p2
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm == 0: return 180.0
        angle = np.degrees(np.arccos(np.clip(dot / norm, -1.0, 1.0)))
        return angle

    def _are_landmarks_visible(self, landmarks, required_indices):
        for index in required_indices:
            if landmarks[index.value].visibility < self.feedback_visibility_threshold:
                return False
        return True

    def evaluate(self, landmarks, current_time):
        if landmarks is None:
            self.feedback_message = "未检测到人体"
        else:
            required = [PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP, PoseLandmark.LEFT_KNEE, PoseLandmark.RIGHT_KNEE,
                        PoseLandmark.LEFT_ANKLE, PoseLandmark.RIGHT_ANKLE]
            if not self._are_landmarks_visible(landmarks, required):
                self.feedback_message = "请确保全身可见"
                self.squat_stage = "stand"
                self.hold_start_time = None
            else:
                l_hip, r_hip = landmarks[PoseLandmark.LEFT_HIP.value], landmarks[PoseLandmark.RIGHT_HIP.value]
                l_knee, r_knee = landmarks[PoseLandmark.LEFT_KNEE.value], landmarks[PoseLandmark.RIGHT_KNEE.value]
                l_ankle, r_ankle = landmarks[PoseLandmark.LEFT_ANKLE.value], landmarks[PoseLandmark.RIGHT_ANKLE.value]

                left_knee_angle = self._calculate_angle(l_hip, l_knee, l_ankle)
                right_knee_angle = self._calculate_angle(r_hip, r_knee, r_ankle)
                avg_knee_angle = (left_knee_angle + right_knee_angle) / 2

                if self.squat_stage == "stand":
                    if avg_knee_angle < 160:  # Start squatting
                        self.squat_stage = "squatting"
                        self.hold_start_time = None
                        self.feedback_message = "慢慢下蹲至膝盖90度"
                elif self.squat_stage == "squatting":
                    is_angle_correct = abs(avg_knee_angle - 90) <= 20  # Wider tolerance for 90 deg
                    if is_angle_correct:
                        if self.hold_start_time is None:
                            self.hold_start_time = current_time
                        self.current_hold_time = current_time - self.hold_start_time
                        if self.current_hold_time >= self.hold_time_min:
                            self.repetition_counter += 1
                            self.feedback_message = f"太棒了！完成一次 {self.hold_time_min} 秒静蹲！"
                            self.squat_stage = "completed"
                        else:
                            self.feedback_message = f"保持住！还剩 {self.hold_time_min - self.current_hold_time:.0f} 秒"
                    else:
                        self.hold_start_time = None  # Angle is wrong, reset timer
                        self.feedback_message = f"膝盖角度: {int(avg_knee_angle)}°，请调整至90度"

                    if avg_knee_angle > 160:  # User stood up
                        self.squat_stage = "stand"
                        self.feedback_message = "动作中断，请重新开始"
                elif self.squat_stage == "completed":
                    if avg_knee_angle > 160:
                        self.squat_stage = "stand"
                        self.feedback_message = "请准备下一次靠墙静蹲"

        metrics = {
            "状态": self.squat_stage,
            "计数": self.repetition_counter,
            "本次计时": f"{self.current_hold_time:.1f}s"
        }
        return self.feedback_message, metrics

    def reset(self):
        self.squat_stage = "stand"
        self.hold_start_time = None
        self.current_hold_time = 0
        self.repetition_counter = 0
        self.feedback_message = "请背靠墙壁准备"
