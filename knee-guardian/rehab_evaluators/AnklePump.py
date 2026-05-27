# rehab_evaluators/AnklePump.py
import numpy as np
import mediapipe as mp

PoseLandmark = mp.solutions.pose.PoseLandmark


class AnklePumpEvaluator:
    """
    评估踝泵运动。
    - 追踪踝关节的屈伸。
    - 状态: neutral -> up (勾脚尖) -> down (踩) -> neutral
    """

    def __init__(self):
        self.stage = "neutral"
        self.counter = 0
        self.feedback = "请放平脚踝，准备开始"

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def evaluate(self, landmarks, current_time):
        try:
            # 使用右脚进行评估
            knee = [landmarks[PoseLandmark.RIGHT_KNEE.value].x, landmarks[PoseLandmark.RIGHT_KNEE.value].y]
            ankle = [landmarks[PoseLandmark.RIGHT_ANKLE.value].x, landmarks[PoseLandmark.RIGHT_ANKLE.value].y]
            foot_index = [landmarks[PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                          landmarks[PoseLandmark.RIGHT_FOOT_INDEX.value].y]

            # 计算脚踝角度 (小腿与脚掌)
            ankle_angle = self.calculate_angle(knee, ankle, foot_index)

            # 评估逻辑: 勾脚尖时角度变小，踩下去时角度变大
            if ankle_angle < 80:  # 勾脚尖
                if self.stage == 'down' or self.stage == 'neutral':
                    self.feedback = "向上勾脚尖"
                self.stage = "up"

            if ankle_angle > 100:  # 踩下去
                if self.stage == 'up':
                    self.counter += 1
                    self.feedback = "向下踩"
                self.stage = "down"

        except Exception as e:
            self.feedback = "未检测到脚部关键点"

        metrics_display = {"循环次数": self.counter}
        return self.feedback, metrics_display

    def reset(self):
        self.stage = "neutral"
        self.counter = 0
        self.feedback = "计数已重置，请重新开始"

