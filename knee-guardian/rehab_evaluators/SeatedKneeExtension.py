# rehab_evaluators/SeatedKneeExtension.py
import numpy as np
import mediapipe as mp

PoseLandmark = mp.solutions.pose.PoseLandmark


class SeatedKneeExtensionEvaluator:
    """
    评估坐姿伸膝动作。
    - 膝关节角度从 < 100° (弯曲) 到 > 160° (伸直) 算一次。
    """

    def __init__(self):
        self.stage = "down"  # 初始状态为膝盖弯曲
        self.counter = 0
        self.feedback = "请坐好，准备开始"

    def calculate_angle(self, a, b, c):
        a = np.array(a)  # First
        b = np.array(b)  # Mid
        c = np.array(c)  # End

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def evaluate(self, landmarks, current_time):
        try:
            hip = [landmarks[PoseLandmark.RIGHT_HIP.value].x, landmarks[PoseLandmark.RIGHT_HIP.value].y]
            knee = [landmarks[PoseLandmark.RIGHT_KNEE.value].x, landmarks[PoseLandmark.RIGHT_KNEE.value].y]
            ankle = [landmarks[PoseLandmark.RIGHT_ANKLE.value].x, landmarks[PoseLandmark.RIGHT_ANKLE.value].y]

            # 计算膝关节角度
            knee_angle = self.calculate_angle(hip, knee, ankle)

            # 评估逻辑
            if knee_angle > 160:
                self.stage = "up"
                self.feedback = "保持伸直"
            if knee_angle < 100 and self.stage == 'up':
                self.stage = "down"
                self.counter += 1
                self.feedback = "缓慢放下"

        except Exception as e:
            self.feedback = "未检测到关键点，请确保身体在画面内"

        metrics_display = {"计数": self.counter}
        return self.feedback, metrics_display

    def reset(self):
        self.stage = "down"
        self.counter = 0
        self.feedback = "计数已重置，请重新开始"
