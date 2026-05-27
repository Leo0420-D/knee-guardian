# rehab_evaluators/one_leg_stand_evaluator.py
import numpy as np
import mediapipe as mp
import time

PoseLandmark = mp.solutions.pose.PoseLandmark


class OneLegStandEvaluator:
    def __init__(self, foot_lift_threshold=0.05, hold_duration=20.0):
        self.foot_lift_threshold = foot_lift_threshold
        self.hold_duration = hold_duration
        self.reset()

    def _detect_support_foot(self, landmarks):
        left_ankle_y = landmarks[PoseLandmark.LEFT_ANKLE.value].y
        right_ankle_y = landmarks[PoseLandmark.RIGHT_ANKLE.value]
        foot_diff = abs(left_ankle_y - right_ankle_y)

        # 通过比较脚踝的y坐标和可见性来判断哪只脚抬起
        if foot_diff > self.foot_lift_threshold:
            if left_ankle_y > right_ankle_y:
                return "left"  # 左脚更低，是支撑脚
            else:
                return "right"  # 右脚更低，是支撑脚
        return "both"  # 双脚着地

    def evaluate(self, landmarks):
        try:
            support_foot = self._detect_support_foot(landmarks)

            # 阶段：双脚着地
            if self.stage == "DOWN":
                if support_foot != "both":
                    self.stage = "UP"
                    self.start_time = time.time()
                    self.feedback = f"开始用{support_foot}脚站立，保持！"
                else:
                    self.feedback = "请抬起一条腿开始单脚站立"

            # 阶段：单脚站立
            elif self.stage == "UP":
                if support_foot == "both":
                    self.feedback = "挑战失败！请重新开始"
                    self.stage = "DOWN"
                    self.start_time = None
                else:
                    elapsed_time = time.time() - self.start_time
                    remaining_time = self.hold_duration - elapsed_time

                    if elapsed_time >= self.hold_duration:
                        self.counter += 1
                        self.feedback = f"太棒了！完成第 {self.counter} 次！"
                        self.stage = "DOWN"
                        self.start_time = None
                    else:
                        self.feedback = f"保持... 剩余 {int(remaining_time) + 1} 秒"

        except Exception as e:
            self.feedback = "无法检测到关键点，请调整姿势"

        return {
            '状态': self.stage,
            '计数': self.counter,
            '反馈': self.feedback
        }

    def reset(self):
        self.stage = "DOWN"
        self.counter = 0
        self.start_time = None
        self.feedback = "计时已重置，请准备开始单脚站立"
