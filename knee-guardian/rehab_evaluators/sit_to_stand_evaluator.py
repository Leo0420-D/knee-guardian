# rehab_evaluators/sit_to_stand_evaluator.py

import numpy as np
import mediapipe as mp

PoseLandmark = mp.solutions.pose.PoseLandmark


class SitToStandEvaluator:
    def __init__(self, stand_threshold=165, sit_threshold=100):
        self.stand_threshold = stand_threshold
        self.sit_threshold = sit_threshold
        self.stage = 'STAND'  # Start in a standing position
        self.counter = 0
        self.feedback = "请准备开始"

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def evaluate(self, landmarks):
        try:
            # 获取左侧身体关键点
            l_shoulder = [landmarks[PoseLandmark.LEFT_SHOULDER.value].x, landmarks[PoseLandmark.LEFT_SHOULDER.value].y]
            l_hip = [landmarks[PoseLandmark.LEFT_HIP.value].x, landmarks[PoseLandmark.LEFT_HIP.value].y]
            l_knee = [landmarks[PoseLandmark.LEFT_KNEE.value].x, landmarks[PoseLandmark.LEFT_KNEE.value].y]

            # 计算左侧髋部角度
            hip_angle = self.calculate_angle(l_shoulder, l_hip, l_knee)

            # 评估逻辑
            if hip_angle > self.stand_threshold:
                if self.stage == 'SIT':
                    self.feedback = "动作完成！"
                self.stage = 'STAND'
            elif hip_angle < self.sit_threshold and self.stage == 'STAND':
                self.stage = 'SIT'
                self.counter += 1
                self.feedback = "请站起"

            if self.stage == 'STAND':
                self.feedback = "请坐下"
            elif self.stage == 'SIT':
                self.feedback = "请站起"

        except Exception as e:
            self.feedback = "无法检测到关键点，请调整姿势"

        return {
            '状态': self.stage,
            '计数': self.counter,
            '反馈': self.feedback
        }

