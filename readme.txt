# 多维膝管家

> 基于X光影像与姿态识别的膝关节骨关节炎智能分级与康复训练系统

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)

---

## 项目结构

```
./
├── Knee/                    # 代表性X光影像集（K-L 0–4级）
│   ├── 0Normal/
│   ├── 1Doubtful/
│   ├── 2Mild/
│   ├── 3Moderate/
│   └── 4Severe/
├── knee-guardian/              # 核心代码模块
│   ├── Fonts/               # 中文字体（SimHei.ttf，用于视频标注）
│   ├── models/              # TensorFlow 预训练模型
│   ├── rehab_evaluators/    # 康复动作评估器（基于MediaPipe姿态识别）
│   ├── static/              # 静态资源（CSS/JS/图片）
│   ├── templates/           # Jinja2 HTML 模板
│   ├── app.py               # Flask 主应用入口
│   ├── spark_client.py      # 讯飞星火大模型 API 客户端
│   ├── _mediapipe_patch.py  # MediaPipe 兼容性补丁
│   └── requirements.txt     # Python 依赖列表
├── .venv/                   # Python 虚拟环境
├── _MACOSX/                 # macOS 元数据（可删除）
└── readme.txt               # 本文件
```

---

## 功能概览

"多维膝管家"是一个面向膝关节骨关节炎（Osteoarthritis, OA）的智能辅助系统，集成四大模块：

### 1. X光影像智能分级
- 基于 TensorFlow 深度学习模型，按 Kellgren-Lawrence (K-L) 标准对膝关节X光片进行 0–4 级自动分类
- 支持 Web 端上传图片，实时返回分级结果与置信度
- 结合 AI 助手（讯飞星火）提供个性化解读与管理建议

### 2. 风险评估问卷
- 涵盖基本信息、膝关节症状、既往病史等多维度问诊
- 自动计算风险评分并划分等级（低/中/高风险）
- 根据风险因素生成个性化预防建议

### 3. AI 智能对话
- 集成讯飞星火大模型（Spark 4.0 Ultra），提供骨科康复理疗师角色对话
- 结合问卷结果或X光诊断上下文，给出专业、个性化的健康指导

### 4. 康复训练评估
- 基于 MediaPipe 实时姿态识别，通过摄像头捕捉用户动作
- 支持 12 种康复运动：深蹲、单腿站立、平板支撑、靠墙静蹲、臀桥、坐位体前屈、肩部前屈上举、颈部旋转、仰卧抬腿、坐姿伸膝、踝泵训练、仰卧直腿抬高
- 实时反馈动作质量、计数与时长等指标

---

## 康复评估器说明

`rehab_evaluators/` 目录包含以下动作评估器：

| 文件 | 对应动作 | 评估指标 |
|------|----------|----------|
| `Squats.py` | 椅子起立/深蹲 | 动作幅度、次数 |
| `onelegstand.py` | 单腿站立 | 平衡时长 |
| `planktimer.py` | 平板支撑 | 支撑时长 |
| `wallsquat.py` | 靠墙静蹲 | 蹲姿角度、时长 |
| `hipbridge.py` | 臀桥 | 抬起次数/时长 |
| `sitandreach.py` | 坐位体前屈 | 前屈幅度 |
| `shoulderflexion.py` | 肩部前屈上举 | 上举角度 |
| `Neckrotation.py` | 颈部旋转 | 旋转角度 |
| `laylagraise.py` | 仰卧直腿抬高 | 抬腿角度、次数 |
| `SeatedKneeExtension.py` | 坐姿伸膝 | 伸膝角度、次数 |
| `AnklePump.py` | 踝泵训练 | 屈伸次数 |
| `one_leg_stand_evaluator.py` | 单腿站立（备选） | 平衡时长 |

---

## 快速启动

### 步骤 1：环境准备

```bash
# 创建并激活虚拟环境
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 安装依赖
pip install -r knee-guardian/requirements.txt
```

### 步骤 2：确保资源就绪

- **字体文件**：确保 `knee-guardian/Fonts/SimHei.ttf` 存在（用于视频反馈中文显示）
- **模型文件**：首次运行时会自动从 Google Drive 下载 `models/model_4_8200.h5`，或手动放置
- **讯飞星火 API**：在 `app.py` 中配置 `SPARK_APPID`、`SPARK_API_KEY`、`SPARK_API_SECRET`

### 步骤 3：启动应用

```bash
cd knee-guardian
python app.py
```

应用默认运行在 **http://127.0.0.1:5000**，支持局域网访问。

### 步骤 4：使用流程

1. 打开浏览器访问首页，选择功能入口
2. **AI 预防建议** → 填写风险评估问卷 → 获取个性化建议
3. **X光诊断** → 上传膝关节X光片 → 查看 AI 分级结果
4. **康复训练** → 选择康复动作 → 开启摄像头 → 实时动作评估与反馈

---

## 技术栈

- **后端**：Flask (Python)
- **深度学习**：TensorFlow (X光分级模型)
- **姿态识别**：MediaPipe Pose
- **AI 对话**：讯飞星火 Spark API (WebSocket)
- **图像处理**：OpenCV, Pillow
- **字体**：SimHei (中文标注)

---

## 注意事项

- TensorFlow >= 2.11 在原生 Windows 上不支持 GPU，需使用 WSL2 或 DirectML 插件
- 摄像头功能需要设备配备可用摄像头
- AI 对话功能依赖讯飞星火 API，需有效的 API 凭证
- 本系统仅供辅助参考，不能替代执业医师的诊断与治疗建议
