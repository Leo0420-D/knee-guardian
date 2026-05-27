# rehab_evaluators/onelegstand.py

import numpy as np
import mediapipe as mp
import time

PoseLandmark = mp.solutions.pose.PoseLandmark


class OneLegStandEvaluator:
    def __init__(self,
                 hold_time_threshold=30,
                 foot_lift_threshold=0.05,
                 feedback_visibility_threshold=0.7
                 ):
        self.hold_time_threshold = hold_time_threshold
        self.foot_lift_threshold = foot_lift_threshold
        self.feedback_visibility_threshold = feedback_visibility_threshold
        self.reset()

    def _are_landmarks_visible(self, landmarks, required_indices):
        for index in required_indices:
            if landmarks[index.value].visibility < self.feedback_visibility_threshold:
                return False
        return True

    def _detect_support_foot(self, landmarks):
        left_ankle = landmarks[PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[PoseLandmark.RIGHT_ANKLE.value]
        foot_diff = left_ankle.y - right_ankle.y

        if abs(foot_diff) > self.foot_lift_threshold:
            return "right" if foot_diff > 0 else "left"
        return None

    def evaluate(self, landmarks, current_time):
        if landmarks is None:
            self.feedback_message = "未检测到人体"
            self.stand_stage = "both_feet"
            self.hold_start_time = None
        else:
            required_landmarks = [
                PoseLandmark.LEFT_ANKLE, PoseLandmark.RIGHT_ANKLE
            ]
            if not self._are_landmarks_visible(landmarks, required_landmarks):
                self.feedback_message = "请确保双脚可见"
                self.stand_stage = "both_feet"
                self.hold_start_time = None
            else:
                detected_foot = self._detect_support_foot(landmarks)

                if self.stand_stage == "both_feet":
                    if detected_foot:
                        self.stand_stage = f"{detected_foot}_stand"
                        self.support_foot = detected_foot
                        self.hold_start_time = current_time
                        self.feedback_message = f"开始用 {'左' if detected_foot == 'left' else '右'} 脚站立"

                elif self.stand_stage.endswith("_stand"):
                    if not detected_foot or detected_foot != self.support_foot:
                        self.feedback_message = "挑战失败！请双脚站稳后重试"
                        self.current_hold_time = current_time - self.hold_start_time
                        if self.current_hold_time > self.session_best:
                            self.session_best = self.current_hold_time
                        self.stand_stage = "both_feet"
                        self.hold_start_time = None
                    else:
                        self.current_hold_time = current_time - self.hold_start_time
                        if self.current_hold_time >= self.hold_time_threshold:
                            self.session_best = self.hold_time_threshold
                            self.completed_trials += 1
                            self.feedback_message = f"太棒了！完成 {self.hold_time_threshold} 秒挑战！"
                            self.stand_stage = "completed"
                        else:
                            self.feedback_message = f"保持稳定！已坚持 {self.current_hold_time:.1f} 秒"

                elif self.stand_stage == "completed":
                    if not detected_foot:
                        self.stand_stage = "both_feet"
                        self.feedback_message = "请准备下一次挑战"

        metrics = {
            "状态": self.stand_stage,
            "当前计时": f"{self.current_hold_time:.1f}s",
            "最佳记录": f"{self.session_best:.1f}s",
            "完成次数": self.completed_trials
        }
        return self.feedback_message, metrics

    def reset(self):
        self.stand_stage = "both_feet"
        self.hold_start_time = None
        self.current_hold_time = 0
        self.completed_trials = 0
        self.session_best = 0
        self.feedback_message = "请抬起一条腿开始单脚站立"
        self.support_foot = None
