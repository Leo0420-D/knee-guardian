# rehab_evaluators/planktimer.py

import numpy as np
import mediapipe as mp
import time

PoseLandmark = mp.solutions.pose.PoseLandmark


class PlankTimer:
    def __init__(self, angle_threshold=160, visibility_threshold=0.7):
        self.angle_threshold = angle_threshold
        self.visibility_threshold = visibility_threshold
        self.reset()

    def _calculate_angle(self, p1, p2, p3):
        p1, p2, p3 = np.array([p1.x, p1.y]), np.array([p2.x, p2.y]), np.array([p3.x, p3.y])
        v1, v2 = p1 - p2, p3 - p2
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        return np.degrees(np.arccos(np.clip(dot / norm, -1.0, 1.0))) if norm != 0 else 180.0

    def evaluate(self, landmarks, current_time):
        required_landmarks = [
            PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER,
            PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP,
            PoseLandmark.LEFT_KNEE, PoseLandmark.RIGHT_KNEE
        ]

        if not all(landmarks[idx.value].visibility > self.visibility_threshold for idx in required_landmarks):
            self.feedback_message = "请确保全身都在摄像头范围内"
            if self.plank_stage == "planking": self.plank_stage = "not_started"; self.start_time = None
        else:
            l_shoulder, r_shoulder = landmarks[PoseLandmark.LEFT_SHOULDER.value], landmarks[
                PoseLandmark.RIGHT_SHOULDER.value]
            l_hip, r_hip = landmarks[PoseLandmark.LEFT_HIP.value], landmarks[PoseLandmark.RIGHT_HIP.value]
            l_knee, r_knee = landmarks[PoseLandmark.LEFT_KNEE.value], landmarks[PoseLandmark.RIGHT_KNEE.value]

            avg_hip_angle = (self._calculate_angle(l_shoulder, l_hip, l_knee) + self._calculate_angle(r_shoulder, r_hip,
                                                                                                      r_knee)) / 2
            is_straight = avg_hip_angle > self.angle_threshold

            if self.plank_stage == "not_started":
                if is_straight:
                    self.plank_stage = "planking"
                    self.start_time = current_time
                    self.feedback_message = "计时开始！保持身体成一条直线"
                else:
                    self.feedback_message = "请抬高臀部，伸直身体"
            elif self.plank_stage == "planking":
                if not is_straight:
                    self.plank_stage = "finished"
                    self.session_best = current_time - self.start_time
                    self.feedback_message = f"挑战结束！坚持了 {self.session_best:.1f} 秒"
                else:
                    self.elapsed_time = current_time - self.start_time
                    self.feedback_message = f"保持住！已坚持 {self.elapsed_time:.1f} 秒"
            elif self.plank_stage == "finished":
                self.feedback_message = f"上次成绩: {self.session_best:.1f} 秒。请重新摆好姿势开始"

        metrics = {
            "状态": self.plank_stage,
            "当前计时": f"{self.elapsed_time:.1f}s",
            "最佳记录": f"{self.session_best:.1f}s"
        }
        return self.feedback_message, metrics

    def reset(self):
        self.plank_stage = "not_started"
        self.start_time = None
        self.elapsed_time = 0
        self.session_best = 0
        self.feedback_message = "请准备开始平板支撑"
