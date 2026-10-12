# app.py

# -----------------------------------------------------------------------------
# 导入必要的库 (这部分保持不变)
# -----------------------------------------------------------------------------
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, session, Response
import os
import cv2
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename
from datetime import timedelta
import gdown
import time
import _mediapipe_patch  # noqa: F401 - must be imported before mediapipe
import mediapipe as mp
import threading
import json
import re
from PIL import Image, ImageDraw, ImageFont

# ... (其他导入保持不变) ...
try:
    from spark_client import SparkChatClient

    print("自定义的讯飞星火客户端 (spark_client.py) 加载成功。")
except ImportError:
    print("!!! 错误: 未找到 'spark_client.py' 文件。请确保该文件与 app.py 在同一目录下。")
    SparkChatClient = None

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
from rehab_evaluators.Squats import SquatEvaluator
from rehab_evaluators.onelegstand import OneLegStandEvaluator
from rehab_evaluators.laylagraise import LayLegRaiseEvaluator
from rehab_evaluators.SeatedKneeExtension import SeatedKneeExtensionEvaluator
from rehab_evaluators.AnklePump import AnklePumpEvaluator
from rehab_evaluators.planktimer import PlankTimer
from rehab_evaluators.wallsquat import WallSquatEvaluator
from rehab_evaluators.hipbridge import HipBridgeTimer
from rehab_evaluators.sitandreach import SitAndReachEvaluator
from rehab_evaluators.shoulderflexion import ShoulderFlexionEvaluator
from rehab_evaluators.Neckrotation import NeckRotationEvaluator

# -----------------------------------------------------------------------------
# Flask 应用初始化与配置 (这部分保持不变)
# -----------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "a_very_secret_and_secure_key_for_dev")
app.permanent_session_lifetime = timedelta(days=7)
UPLOAD_FOLDER = "static/img/images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = "models/model_4_8200.h5"
os.makedirs("models", exist_ok=True)

# -----------------------------------------------------------------------------
# 资源加载 (这部分保持不变)
# -----------------------------------------------------------------------------
try:
    FONT_PATH = "Fonts/SimHei.ttf"
    if not os.path.exists(FONT_PATH):
        raise FileNotFoundError(f"字体文件未找到: {FONT_PATH}。请下载并放置在正确位置。")
    FONT = ImageFont.truetype(FONT_PATH, 24)
    print(f"中文字体 '{FONT_PATH}' 加载成功。")
except Exception as e:
    print(f"!!! 警告: 加载中文字体失败: {e}。视频反馈文字将无法正常显示。")
    FONT = None

if not os.path.exists(MODEL_PATH):
    print(f"Downloading {MODEL_PATH}...")
    gdown.download(id="1wTHYfDJAr4ClcOvv5u9XqHVlB9t3Woiv", output=MODEL_PATH, quiet=False)

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("X-Ray TensorFlow model loaded successfully.")
except Exception as e:
    print(f"Error loading TensorFlow model: {e}")
    model = None

# -----------------------------------------------------------------------------
# 讯飞星火 API 设置
#
# 凭据一律从环境变量读取，不要硬编码进源码 —— 本仓库是公开的。
# 配置方式（二选一）：
#   本地：复制 knee-guardian/.env.example 为 knee-guardian/.env，填入自己的凭据
#   部署：在平台的环境变量面板里设置同名变量
# 未配置凭据时，AI 对话功能自动降级禁用，其余功能不受影响。
# -----------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    load_dotenv()  # 读取当前工作目录下的 .env（已被 .gitignore 忽略，不会进仓库）
except ImportError:
    pass  # 未安装 python-dotenv 时退化为只读系统环境变量

SPARK_APPID = os.environ.get("SPARK_APPID", "")
SPARK_API_KEY = os.environ.get("SPARK_API_KEY", "")
SPARK_API_SECRET = os.environ.get("SPARK_API_SECRET", "")
SPARK_URL = os.environ.get("SPARK_URL", "wss://spark-api.xf-yun.com/v4.0/chat")
SPARK_DOMAIN = os.environ.get("SPARK_DOMAIN", "4.0Ultra")

SPARK_CREDENTIALS_READY = all((SPARK_APPID, SPARK_API_KEY, SPARK_API_SECRET))
spark_client = None

if SparkChatClient and SPARK_CREDENTIALS_READY:
    try:
        spark_client = SparkChatClient(
            app_id=SPARK_APPID, api_key=SPARK_API_KEY, api_secret=SPARK_API_SECRET,
            spark_url=SPARK_URL, domain=SPARK_DOMAIN
        )
        print("自定义的讯飞星火客户端初始化成功。")
    except Exception as e:
        print(f"!!! 错误: 自定义的讯飞星火客户端初始化失败: {e}")
elif SparkChatClient:
    print("!!! 警告: 未检测到讯飞星火凭据，AI 对话功能已禁用（其余功能不受影响）。")
    print("     请设置环境变量 SPARK_APPID / SPARK_API_KEY / SPARK_API_SECRET，")
    print("     或复制 knee-guardian/.env.example 为 knee-guardian/.env 并填入凭据。")
else:
    print("!!! 警告: 自定义的讯飞星火客户端未加载，AI聊天功能将不可用。")

# -----------------------------------------------------------------------------
# 应用程序数据结构 (所有数据结构保持不变)
# -----------------------------------------------------------------------------
EXERCISE_CONFIG = {
    # ... (这部分数据完全不变) ...
    "kl_0_1": {"grade_name": "K-L 0-I 级 (早期 / 预防期)",
               "description": "核心目标：预防疾病进展，增强肌肉力量与身体本体感觉，维持关节稳定性。", "exercises": {
            "one_leg_stand": {"name": "单腿站立",
                              "description": "单腿站立是提高平衡能力和本体感觉的绝佳训练。它能有效锻炼膝、踝关节周围的稳定肌群，对于预防运动损伤和跌倒风险非常有帮助。初期可扶墙或椅子进行。",
                              "evaluator_class": OneLegStandEvaluator}, "seated_knee_extension": {"name": "坐姿伸膝",
                                                                                                  "description": "此动作为无负重下的股四头肌强化训练。坐在椅子上，缓慢伸直膝关节至水平，保持3秒后缓慢放下。它能有效增强大腿前侧肌肉力量，提升关节应力耐受性。",
                                                                                                  "evaluator_class": SeatedKneeExtensionEvaluator}}},
    "kl_2": {"grade_name": "K-L II 级 (轻度)",
             "description": "核心目标：在控制疼痛的前提下，进行功能性训练，改善关节的稳定性。", "exercises": {
            "chair_stand": {"name": "椅子起立",
                            "description": "椅子起立是功能性下肢力量训练，模拟日常生活中最常见的动作。它能强化完成“站起-坐下”所需的股四头肌和臀肌力量。请保持动作平稳有控制。",
                            "evaluator_class": SquatEvaluator}}},
    "kl_3": {"grade_name": "K-L III 级 (中度)",
             "description": "核心目标：保护关节，避免过度磨损，维持基础的肌肉功能和关节活动度。", "exercises": {
            "lay_leg_raise": {"name": "仰卧直腿抬高",
                              "description": "在仰卧位进行直腿抬高，是一种避免膝关节屈伸的等长肌力训练。膝下可垫毛巾卷，绷紧大腿肌肉，将腿抬高至屈髋30°左右，能有效维持股四头肌功能，减少软骨磨损。",
                              "evaluator_class": LayLegRaiseEvaluator}}},
    "kl_4": {"grade_name": "K-L IV 级 (重度)",
             "description": "核心目标：进行床上或辅助下的轻量活动，管理疼痛，预防深静脉血栓等并发症。", "exercises": {
            "ankle_pump": {"name": "踝泵训练",
                           "description": "踝泵是最基础的床上运动，通过缓慢屈伸踝关节至极限，促进下肢血液循环，是预防深静脉血栓的关键措施。请将动作做到位，感受小腿肌肉的收缩与拉伸。",
                           "evaluator_class": AnklePumpEvaluator}}},
    "other_exercises": {"grade_name": "其他运动",
                        "description": "这些运动可作为日常康复训练的补充，有助于增强肌肉力量和灵活性。", "exercises": {
            "plank": {"name": "平板支撑",
                      "description": "平板支撑是一种全身性的核心稳定性训练，能有效增强腹部、背部和臀部肌肉力量，减轻膝关节压力。",
                      "evaluator_class": PlankTimer}, "wall_squat": {"name": "靠墙静蹲",
                                                                     "description": "背部靠墙，双脚与肩同宽，缓慢下蹲至大腿与地面平行，保持一段时间。能有效锻炼大腿前侧和臀部肌肉，减轻膝关节负担。",
                                                                     "evaluator_class": WallSquatEvaluator},
            "hip_bridge": {"name": "臀桥",
                           "description": "仰卧位，双腿屈膝，双脚平放在地面，缓慢抬起臀部至肩、髋、膝成一条直线。能有效锻炼臀部和大腿后侧肌肉。",
                           "evaluator_class": HipBridgeTimer}, "sit_and_reach": {"name": "坐位体前屈",
                                                                                 "description": "坐在地上，双腿伸直，缓慢向前弯腰，尽量用手触摸脚尖。能有效拉伸大腿后侧和腰部肌肉，提高关节灵活性。",
                                                                                 "evaluator_class": SitAndReachEvaluator},
            "shoulder_flexion": {"name": "肩部前屈上举",
                                 "description": "站立位，双臂自然下垂，缓慢向上抬起双臂至头顶上方，感受肩部肌肉的伸展和收缩。能有效锻炼肩部肌肉，提高肩部的活动度。",
                                 "evaluator_class": ShoulderFlexionEvaluator}, "neck_rotation": {"name": "颈部旋转",
                                                                                                 "description": "坐在椅子上，保持身体正直，缓慢向左或向右转头，感受颈部肌肉的拉伸。能有效缓解颈部肌肉紧张，预防颈椎病。",
                                                                                                 "evaluator_class": NeckRotationEvaluator},
            "lay_leg_raise_new": {"name": "仰卧抬腿",
                                  "description": "仰卧位，双腿伸直，缓慢向上抬起双腿至与地面垂直，然后缓慢放下。能有效锻炼腹部和腿部肌肉，增强核心力量。",
                                  "evaluator_class": LayLegRaiseEvaluator}}}
}
questions_data = [
    # ... (这部分数据完全不变) ...
    {'title': '第一部分：基本信息与生活习惯', 'note': '这部分信息有助于我们了解您的基础风险暴露情况。', 'questions': [
        {'id': 'age', 'type': 'select', 'text': '您的年龄段是？',
         'options': [{'text': '40岁以下', 'score': 0}, {'text': '40-55岁', 'score': 1}, {'text': '56-65岁', 'score': 2},
                     {'text': '65岁以上', 'score': 3}]},
        {'id': 'gender', 'type': 'select', 'text': '您的生理性别是？',
         'options': [{'text': '男性', 'score': 1}, {'text': '女性', 'score': 2}]},
        {'id': 'bmi', 'type': 'bmi_input', 'text': '请输入您的身高与体重以计算BMI：'},
        {'id': 'occupation', 'type': 'select', 'text': '您的职业或日常活动是否需要长时间站立、行走或搬运重物？',
         'options': [{'text': '几乎不需要', 'score': 0}, {'text': '偶尔需要', 'score': 1},
                     {'text': '经常需要', 'score': 2}, {'text': '几乎总是', 'score': 3}]}
    ]},
    {'title': '第二部分：膝关节症状与感受', 'note': '这部分关注您的主观感受，是评估膝关节健康状况的重要依据。',
     'questions': [
         {'id': 'pain_freq', 'type': 'select', 'text': '您多久会感到膝盖疼痛？',
          'options': [{'text': '从不', 'score': 0}, {'text': '每月几次', 'score': 1}, {'text': '每周几次', 'score': 2},
                      {'text': '几乎每天', 'score': 3}]},
         {'id': 'stiffness', 'type': 'select', 'text': '早上起床或久坐后，您是否感到膝盖僵硬，需要活动一下才能缓解？',
          'options': [{'text': '没有', 'score': 0}, {'text': '偶尔有，持续时间很短', 'score': 1},
                      {'text': '经常有，持续几分钟', 'score': 2}, {'text': '每天都有，且持续较长时间', 'score': 3}]},
         {'id': 'activity_pain', 'type': 'select', 'text': '上下楼梯、下蹲或走路时，您的膝盖疼痛会加重吗？',
          'options': [{'text': '不会', 'score': 0}, {'text': '轻微加重', 'score': 1}, {'text': '明显加重', 'score': 2},
                      {'text': '非常疼痛，导致活动困难', 'score': 3}]}
     ]},
    {'title': '第三部分：既往病史与运动习惯', 'note': '了解您的过去和现在，有助于全面评估未来风险。', 'questions': [
        {'id': 'injury_history', 'type': 'select', 'text': '您的膝关节（包括韧带、半月板）是否受过伤？',
         'options': [{'text': '没有', 'score': 0}, {'text': '有轻微扭伤史，已完全恢复', 'score': 1},
                     {'text': '有较严重的伤病史（如韧带撕裂、半月板损伤）', 'score': 3}]},
        {'id': 'exercise_habit', 'type': 'select', 'text': '您是否有规律的体育锻炼习惯（每周至少3次，每次30分钟）？',
         'options': [{'text': '有，且包含力量训练', 'score': 0}, {'text': '有，但主要是散步等低强度运动', 'score': 1},
                     {'text': '没有规律的锻炼习惯', 'score': 2}]}
    ]}
]

# -----------------------------------------------------------------------------
# 辅助变量和函数 (大部分保持不变)
# -----------------------------------------------------------------------------
camera = None
camera_lock = threading.Lock()
exercise_status_data = {}
exercise_status_lock = threading.Lock()

flat_exercise_config = {}
for grade_key, grade_data in EXERCISE_CONFIG.items():
    if 'exercises' in grade_data and grade_data['exercises']:
        for ex_key, ex_data in grade_data['exercises'].items():
            # 为每个 exercise 创建一个新的评估器实例
            ex_data['instance'] = ex_data['evaluator_class']()
            flat_exercise_config[ex_key] = ex_data


def get_spark_response(prompt):
    if not spark_client: return "抱歉，AI助手服务未成功初始化，请检查后台日志。"
    try:
        response = spark_client.get_response(prompt)
        return response
    except Exception as e:
        print(f"Error in get_spark_response using custom client: {e}")
        return f"调用AI服务时发生错误，请稍后重试。错误详情: {e}"


def markdown_to_html(text):
    """将AI生成的Markdown文本转换为HTML，逐行解析，正确处理标题/粗体/列表/段落"""
    lines = text.split('\n')
    result = []
    in_list = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_list:
                result.append('</ul>')
                in_list = False
            continue

        # 粗体: **text** → <strong>text</strong>
        stripped = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', stripped)

        # 标题
        if stripped.startswith('### '):
            if in_list:
                result.append('</ul>')
                in_list = False
            result.append(f'<h3>{stripped[4:]}</h3>')
        elif stripped.startswith('## '):
            if in_list:
                result.append('</ul>')
                in_list = False
            result.append(f'<h2>{stripped[2:]}</h2>')
        elif stripped.startswith('# '):
            if in_list:
                result.append('</ul>')
                in_list = False
            result.append(f'<h1>{stripped[2:]}</h1>')
        # 无序列表: - 或 * 开头
        elif re.match(r'^[-*]\s+', stripped):
            if not in_list:
                result.append('<ul>')
                in_list = True
            result.append(f'<li>{re.sub(r"^[-*]\s+", "", stripped)}</li>')
        # 有序列表: 1. 或 1、 开头
        elif re.match(r'^\d+[\.\、]\s*', stripped):
            if not in_list:
                result.append('<ul>')
                in_list = True
            result.append(f'<li>{re.sub(r"^\d+[\.\、]\s*", "", stripped)}</li>')
        else:
            if in_list:
                result.append('</ul>')
                in_list = False
            result.append(f'<p>{stripped}</p>')

    if in_list:
        result.append('</ul>')

    return '\n'.join(result)


def find_option_text(question_id, value):
    # ... (此函数保持不变) ...
    value = str(value)
    for part in questions_data:
        for q in part['questions']:
            if q['id'] == question_id and q.get('type') == 'select':
                for option in q.get('options', []):
                    if str(option['score']) == value:
                        return option['text']
    return "未知选项"


def get_camera():
    # ... (此函数保持不变) ...
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            print("尝试打开摄像头...")
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                print("无法打开摄像头")
                camera = None
                return None
            else:
                print("摄像头已成功打开。")
        return camera


def release_camera():
    # ... (此函数保持不变) ...
    global camera
    with camera_lock:
        if camera is not None and camera.isOpened():
            camera.release()
            camera = None
            print("摄像头已释放")


# -----------------------------------------------------------------------------
# Flask 路由定义
# -----------------------------------------------------------------------------
@app.route('/')
def intro():
    # ... (此路由保持不变) ...
    release_camera()
    return render_template('intro.html')


@app.route('/prevention')
def prevention_options():
    # ... (此路由保持不变) ...
    release_camera()
    return render_template('prevention_options.html')


@app.route('/diagnosis_rehab')
def diagnosis_rehab_options():
    # ... (此路由保持不变) ...
    release_camera()
    return render_template('rehab_index.html', exercises_by_grade=EXERCISE_CONFIG)


@app.route('/preventive_ai_measures')
def preventive_ai_measures():
    release_camera()
    if 'for_chat_context' not in session or session['for_chat_context'].get('type') != 'questionnaire':
        flash("请先完成风险评估问卷，以便获得个性化的预防建议。", "warning")
        return redirect(url_for('questionnaire'))
    try:
        context = session['for_chat_context']
        high_risk_items = [item for item in context['q_and_a'] if item.get('score', 0) > 1]

        if not high_risk_items:
            risk_text = "该用户膝关节风险评估得分很低，没有明显风险项。"
        else:
            risk_factors = "\n".join(
                [f"- {item['question']}\n  用户回答: {item['answer']}（风险分: {item['score']}）" for item in high_risk_items])
            risk_text = f"用户整体风险等级: {context['risk_level']}（总分: {context['total_score']}）\n主要风险因素:\n{risk_factors}"

        system_prompt = (
            "你是一名资深的骨科康复理疗师，擅长为膝关节问题患者撰写清晰、专业的个性化康复建议报告。"
            "请严格遵循以下格式要求输出：\n"
            "1. 使用 ## 和 ### 作为标题层级\n"
            "2. 使用 - 开头作为分点列表\n"
            "3. 使用 **文字** 标记重点（仅用于关键术语和警告）\n"
            "4. 每个要点控制在2-3句话内，段落之间用空行分隔\n"
            "5. 全文使用简体中文，不要混入英文\n"
            "6. 报告结构：先总体评价，再分项建议（减重/运动/饮食/日常保护），最后给出注意事项")

        user_message = (
            f"请根据以下用户风险评估数据，生成一份膝关节预防与养护建议报告：\n\n{risk_text}\n\n"
            "请开始生成报告。")

        ai_advice = spark_client.chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ])

        ai_advice_html = markdown_to_html(ai_advice)
        chat_context_data = session.get('for_chat_context', None)
        chat_context_json = json.dumps(chat_context_data) if chat_context_data else 'null'
        return render_template('preventive_ai_measures.html', advice=ai_advice_html, chat_context=chat_context_json)
    except Exception as e:
        print(f"An unexpected error occurred in /preventive_ai_measures: {e}")
        flash("生成建议时发生未知错误。", "error")
        return redirect(url_for('result'))


@app.route('/questionnaire', methods=['GET', 'POST'])
def questionnaire():
    # ... (此路由保持不变) ...
    release_camera()
    if request.method == 'POST':
        total_score = 0
        q_and_a_list = []
        form_data = request.form
        for part in questions_data:
            for q in part['questions']:
                question_id = q['id']
                if q.get('type') == 'bmi_input':
                    try:
                        height_cm = float(form_data.get('height'))
                        weight_kg = float(form_data.get('weight'))
                        if height_cm > 0 and weight_kg > 0:
                            height_m = height_cm / 100.0
                            bmi = weight_kg / (height_m ** 2)
                            bmi_score = 0
                            bmi_category = "正常"
                            if bmi >= 28:
                                bmi_score = 3
                                bmi_category = "肥胖"
                            elif bmi >= 24:
                                bmi_score = 2
                                bmi_category = "超重"
                            total_score += bmi_score
                            q_and_a_list.append({
                                'question': '体重指数(BMI)',
                                'answer': f'{bmi:.1f} ({bmi_category})',
                                'score': bmi_score
                            })
                    except (ValueError, TypeError):
                        q_and_a_list.append({
                            'question': '体重指数(BMI)',
                            'answer': '输入无效',
                            'score': 0
                        })
                    continue
                value = form_data.get(question_id)
                if value is not None:
                    score = int(value)
                    total_score += score
                    q_and_a_list.append({
                        'question': q['text'],
                        'answer': find_option_text(question_id, value),
                        'score': score
                    })
        risk_level = '低风险' if total_score <= 5 else ('中等风险' if 6 <= total_score <= 12 else '高风险')
        session['total_score'] = total_score
        session['risk_level'] = risk_level
        session['for_chat_context'] = {'type': 'questionnaire', 'total_score': total_score, 'risk_level': risk_level,
                                       'q_and_a': q_and_a_list}
        session.permanent = True
        return redirect(url_for('result'))
    return render_template('questionnaire.html', questions_data=questions_data)


@app.route('/result')
def result():
    # ... (此路由保持不变) ...
    release_camera()
    total_score = session.get('total_score', 0)
    risk_level = session.get('risk_level', '未知')
    return render_template('result.html', total_score=total_score,
                           risk_level=risk_level)


@app.route('/about_us')
def about_us():
    # ... (此路由保持不变) ...
    release_camera()
    return render_template('about_us.html')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None: raise ValueError("Failed to read image.")
    image = cv2.resize(image, (200, 200))
    image = image / 255.0
    image = np.expand_dims(image, axis=(0, -1))
    return image


# <<< MODIFIED SECTION START: /prediction route >>>
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    release_camera()
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': '未选择文件'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '文件名为空'}), 400
        if file and allowed_file(file.filename):
            try:
                if model is None:
                    raise ValueError("TensorFlow model is not loaded.")

                npimg = np.frombuffer(file.read(), np.uint8)
                image_gray = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
                if image_gray is None:
                    raise ValueError("无法解码图像文件。")

                processed_image = np.expand_dims(cv2.resize(image_gray, (200, 200)) / 255.0, axis=(0, -1))
                predictions = model.predict(processed_image)[0]

                # 内部使用的英文标签
                class_labels_en = ['Normal', 'Doubtful', 'Mild', 'Moderate', 'Severe']

                # 【【【 新增：用于翻译的字典 】】】
                severity_translation = {
                    'Normal': '正常',
                    'Doubtful': '可疑',
                    'Mild': '轻度',
                    'Moderate': '中度',
                    'Severe': '重度'
                }

                kl_grade_map = {'Normal': 0, 'Doubtful': 1, 'Mild': 2, 'Moderate': 3, 'Severe': 4}

                predicted_class_index = np.argmax(predictions)

                # 1. 获取英文标签
                severity_label_en = class_labels_en[predicted_class_index]
                confidence = float(predictions[predicted_class_index])

                # 2. 【【【 翻译成中文标签 】】】
                severity_label_zh = severity_translation.get(severity_label_en, "未知")

                # 3. 构建上下文和返回数据，使用中文标签
                kl_grade = kl_grade_map.get(severity_label_en)

                chat_context = {
                    'type': 'xray',
                    'prediction_result': {
                        'kl_grade': kl_grade,
                        'severity_label': severity_label_zh,  # 在上下文中也使用中文
                        'confidence': f"{confidence:.2%}"
                    }
                }
                session['for_chat_context'] = chat_context
                session.permanent = True

                # 4. 【【【 返回给前端的JSON中使用中文标签 】】】
                return jsonify({
                    'kl_grade': kl_grade,
                    'severity_label': severity_label_zh,  # 此处修改为中文
                    'confidence': f"{confidence:.2%}",
                    'chat_context': chat_context
                })

            except Exception as e:
                print(f"Prediction error: {e}")
                return jsonify({'error': f'诊断出错: {str(e)}'}), 500
        return jsonify({'error': '文件类型不被允许'}), 400

    return render_template('prediction.html')


# <<< MODIFIED SECTION END >>>

@app.route('/rehab/<exercise_key>')
def rehab_exercise_page(exercise_key):
    # ... (此路由保持不变) ...
    if exercise_key not in flat_exercise_config:
        flash("无效的康复运动项目", "error")
        return redirect(url_for('diagnosis_rehab_options'))

    # 每次进入页面时重置该运动的状态
    evaluator = flat_exercise_config[exercise_key].get("instance")
    if evaluator and hasattr(evaluator, 'reset'):
        evaluator.reset()

    with exercise_status_lock:
        exercise_status_data[exercise_key] = {"feedback": "准备开始...", "metrics": {}}

    return render_template('rehab_exercise.html', exercise_key=exercise_key,
                           exercise=flat_exercise_config[exercise_key])


def draw_landmarks_on_image(image, results):
    # ... (此函数保持不变) ...
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
    return image


def display_feedback_on_image(image, feedback_text, position=(10, 30)):
    # ... (此函数保持不变) ...
    if FONT is None:
        cv2.putText(image, "FONT_NOT_LOADED", position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        return image
    pil_im = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_im)
    y = position[1]
    for line in feedback_text.split('|'):
        line = line.strip()
        text_bbox = draw.textbbox((position[0], y), line, font=FONT)
        padding = 5
        bg_rect = (text_bbox[0] - padding, text_bbox[1] - padding, text_bbox[2] + padding, text_bbox[3] + padding)
        draw.rectangle(bg_rect, fill=(0, 0, 0, 128))
        draw.text((position[0], y), line, font=FONT, fill=(255, 255, 255))
        y += text_bbox[3] - text_bbox[1] + 10
    return cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)


def generate_frames(exercise_key):
    # ... (此函数保持不变) ...
    cam = get_camera()
    if not cam:
        error_img = np.zeros((480, 640, 3), dtype=np.uint8)
        error_img = display_feedback_on_image(error_img, "摄像头错误，请检查设备连接")
        _, buffer = cv2.imencode('.jpg', error_img)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        return

    evaluator = flat_exercise_config[exercise_key]["instance"]

    # 确保进入时重置状态
    if hasattr(evaluator, 'reset'):
        evaluator.reset()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            with camera_lock:
                if camera is None or not camera.isOpened():
                    break
                success, frame = camera.read()
            if not success:
                time.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame = draw_landmarks_on_image(frame, results)

            feedback_msg, metrics_display = "请开始动作", {}
            if results.pose_landmarks:
                feedback_msg, metrics_display = evaluator.evaluate(results.pose_landmarks.landmark, time.time())
            else:
                feedback_msg = "未检测到身体姿态，请确保您在摄像头画面内。"

            with exercise_status_lock:
                exercise_status_data[exercise_key] = {"feedback": feedback_msg, "metrics": metrics_display}

            metrics_str = " | ".join([f"{k}: {v}" for k, v in metrics_display.items()])
            frame = display_feedback_on_image(frame,
                                              f"{feedback_msg} ({metrics_str})" if metrics_display else feedback_msg)

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    release_camera()


@app.route('/video_feed/<exercise_key>')
def video_feed(exercise_key):
    # ... (此路由保持不变) ...
    if exercise_key not in flat_exercise_config:
        return "Invalid exercise", 404
    return Response(generate_frames(exercise_key), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/rehab_status/<exercise_key>')
def rehab_status(exercise_key):
    # ... (此路由保持不变) ...
    if exercise_key not in flat_exercise_config:
        return jsonify({"error": "Invalid exercise"}), 404
    with exercise_status_lock:
        return jsonify(exercise_status_data.get(exercise_key, {"feedback": "等待数据...", "metrics": {}}))


@app.route('/rehab_reset/<exercise_key>', methods=['POST'])
def rehab_reset(exercise_key):
    # ... (此路由保持不变) ...
    if exercise_key not in flat_exercise_config:
        return jsonify({"error": "Invalid exercise"}), 404
    evaluator = flat_exercise_config[exercise_key]["instance"]
    if hasattr(evaluator, 'reset'):
        evaluator.reset()
    with exercise_status_lock:
        exercise_status_data[exercise_key] = {"feedback": "计数已重置，请重新开始。", "metrics": {}}
    return jsonify({"message": f"{flat_exercise_config[exercise_key]['name']} 计数器已重置"})


@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    # ... (此路由保持不变) ...
    release_camera()
    return jsonify({"status": "success", "message": "Camera released."})


MAX_HISTORY = 10  # 保留最近 N 轮对话


def build_system_prompt(context):
    """根据上下文生成系统提示词，仅在对话开始时调用一次"""
    if not context or not isinstance(context, dict):
        return "你是一名专业的骨科康复理疗师，请用简体中文回答所有问题。语气专业、亲切、鼓励。"
    if context.get('type') == 'questionnaire':
        q_and_a_text = "\n".join(
            [f"- {item['question']} (回答: {item['answer']})" for item in context.get('q_and_a', [])])
        return (
            f"你是一名专业的骨科康复理疗师。用户刚完成了膝关节风险评估问卷，以下是背景信息：\n"
            f"风险等级: {context.get('risk_level', '未知')}（总分: {context.get('total_score', 0)}）\n"
            f"用户回答摘要:\n{q_and_a_text}\n"
            f"请基于以上信息，用简体中文专业、亲切地回答用户的任何问题。"
            f"你的建议不能替代专业医疗诊断。")
    if context.get('type') == 'xray':
        pred = context.get('prediction_result', {})
        return (
            f"你是一名专业的骨科医生。用户刚上传了膝关节X光片，AI辅助诊断结果如下：\n"
            f"KL分级: {pred.get('kl_grade', '未知')}（{pred.get('severity_label', '未知')}），置信度: {pred.get('confidence', '未知')}\n"
            f"请基于以上信息，用简体中文专业、谨慎地回答用户的问题，解释分级含义并提供初步管理建议。"
            f"注意：你的回答不能替代执业医师的最终诊断。")
    return "你是一名专业的骨科康复理疗师，请用简体中文回答所有问题。语气专业、亲切、鼓励。"


@app.route('/chat_with_spark', methods=['POST'])
def chat_with_spark():
    data = request.get_json()
    user_message = data.get('message')
    context = data.get('context') or session.get('for_chat_context')

    if isinstance(context, str):
        try:
            context = json.loads(context)
        except (json.JSONDecodeError, TypeError):
            context = None

    if not user_message:
        return jsonify({'error': 'Message is required'}), 400

    # 初始化或检测上下文切换，重建对话历史
    context_key = json.dumps(context, sort_keys=True, ensure_ascii=False) if context else "none"
    if 'chat_context_key' not in session or session.get('chat_context_key') != context_key:
        session['chat_history'] = []
        session['chat_context_key'] = context_key
        system_prompt = build_system_prompt(context)
        session['chat_history'].append({"role": "system", "content": system_prompt})

    # 追加用户消息
    session['chat_history'].append({"role": "user", "content": user_message})

    # 裁剪历史：保留 system prompt + 最近 MAX_HISTORY*2 条（user+assistant）
    history = session['chat_history']
    if len(history) > 1 + MAX_HISTORY * 2:
        history = [history[0]] + history[-(MAX_HISTORY * 2):]
        session['chat_history'] = history

    try:
        reply = spark_client.chat(history)
        # 追加 AI 回复到历史
        session['chat_history'].append({"role": "assistant", "content": reply})
        session.modified = True
        return jsonify({'reply': reply})
    except Exception as e:
        print(f"Spark API request failed: {e}")
        # 移除失败的用户消息，避免污染历史
        if session['chat_history'] and session['chat_history'][-1]["role"] == "user":
            session['chat_history'].pop()
        return jsonify({'error': '无法连接到智能助手服务，请稍后再试。'}), 503


# -----------------------------------------------------------------------------
# 程序主入口 (保持不变)
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    print("---------------------------------------------------------")
    print("           守护“膝”望 - 智能康复助手 启动中...           ")
    print("---------------------------------------------------------")
    print(">>> 正在使用自定义的讯飞星火客户端 (spark_client.py)。")
    print(">>> 确保你的讯飞星火API凭证已正确填写在app.py文件中。")
    if not spark_client:
        print("!!! 严重警告: AI聊天功能将无法使用，请检查API凭证和spark_client.py文件。")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True, threaded=True)
