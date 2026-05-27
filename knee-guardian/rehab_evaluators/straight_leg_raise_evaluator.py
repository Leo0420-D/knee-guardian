# rehab_evaluators/straight_leg_raise_evaluator.py
import numpy as np
import mediapipe as mp
import time

PoseLandmark = mp.solutions.pose.PoseLandmark

class StraightLegRaiseEvaluator:
    def __init__(self, leg_straight_threshold=160, hip_angle_up_threshold=25, hold_duration=3.0):
        self.leg_straight_threshold = leg_straight_threshold
        self.hip_angle_up_threshold = hip_angle_up_threshold
        self.hold_duration = hold_duration
        self.reset()

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return angle if angle <= 180 else 360 - angle

    def evaluate(self, landmarks):
        try:
            l_shoulder = [landmarks[PoseLandmark.LEFT_SHOULDER.value].x, landmarks[PoseLandmark.LEFT_SHOULDER.value].y]
            l_hip = [landmarks[PoseLandmark.LEFT_HIP.value].x, landmarks[PoseLandmark.LEFT_HIP.value].y]
            l_knee = [landmarks[PoseLandmark.LEFT_KNEE.value].x, landmarks[PoseLandmark.LEFT_KNEE.value].y]
            l_ankle = [landmarks[PoseLandmark.LEFT_ANKLE.value].x, landmarks[PoseLandmark.LEFT_ANKLE.value].y]

            knee_angle = self.calculate_angle(l_hip, l_knee, l_ankle)
            hip_angle = self.calculate_angle(l_shoulder, l_hip, l_knee)

            if knee_angle < self.leg_straight_threshold:
                self.feedback = "请保持膝盖伸直！"
                if self.stage == 'UP':
                    self.stage = 'DOWN'
                    self.start_time = None
            else:
                if self.stage == 'DOWN':
                    if hip_angle > self.hip_angle_up_threshold:
                        self.stage = 'UP'
                        self.start_time = time.time()
                        self.feedback = "保持住！"
                    else:
                        self.feedback = "请缓慢抬起左腿"
                elif self.stage == 'UP':
                    elapsed_time = time.time() - self.start_time
                    if elapsed_time >= self.hold_duration:
                        self.counter += 1
                        self.feedback = f"完成第 {self.counter} 次！请缓慢放下"
                        self.stage = 'DOWN'
                        self.start_time = None
                    else:
                        remaining_time = self.hold_duration - elapsed_time
                        self.feedback = f"保持... 剩余 {int(remaining_time) + 1} 秒"
        except Exception:
            self.feedback = "未检测到关键点，请确保身体在镜头内"

        return {'状态': self.stage, '计数': self.counter, '反馈': self.feedback}

    def reset(self):
        self.stage = 'DOWN'
        self.counter = 0
        self.start_time = None
        self.feedback = "已重置，请准备开始"
