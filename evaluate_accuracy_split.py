import os
import torch
import torchaudio
import numpy as np
import models
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix

# ==================== 配置区 ====================
# 1. 四个测试文件夹所在的根目录
TEST_ROOT_DIR = r"D:\vscode\ceshi\test100-4"  

# 2. 模型权重路径
CHECKPOINT_PATH = "./experiments/my_audiotransformer_mini/audiotransformer_small/2026-06-04_17-23_08f270585ff711f189c9d8459082dc2d/best_checkpoint_9_0.9859.pt"

# 3. 文件夹名到模型标签索引的映射（严格对应 LABEL_MAP）
FOLDER_TO_IDX = {
    "niao0915": 0,
    "shi1107": 1,
    "lei1104": 2,
    "Noise1106": 3
}

LABEL_MAP = {0: "鸟声", 1: "施工作业声", 2: "雷声", 3: "其他声音(Noise)"}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 音频切片与采样率相关参数
TARGET_SAMPLE_RATE = 16000
SEGMENT_SECONDS = 10
SEGMENT_SAMPLES = TARGET_SAMPLE_RATE * SEGMENT_SECONDS  # 160000 个采样点
MAX_SEGMENTS = 3                                 # 30秒音频刚好切3段

# ==================== 1. 初始化并加载模型 ====================
print("正在加载模型与权重...")
model = models.audiotransformer_small(num_classes=4, pretrained=False)
ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
model.load_state_dict(ckpt['model'])
model = model.to(DEVICE).eval()

# ==================== 2. 批量扫描音频与分段推理 ====================
y_true = []
y_pred = []

print("开始批量扫描音频并进行 10 秒分段切片独立推理...")
for folder_name, true_label in FOLDER_TO_IDX.items():
    folder_path = os.path.join(TEST_ROOT_DIR, folder_name)
    
    if not os.path.exists(folder_path):
        print(f"警告: 文件夹不存在，跳过 -> {folder_path}")
        continue
        
    wav_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.wav')]
    print(f"正在处理文件夹 [{folder_name}]，共发现 {len(wav_files)} 个音频...")
    
    for wav_name in tqdm(wav_files, desc=f"推理 {folder_name}"):
        wav_path = os.path.join(folder_path, wav_name)
        
        try:
            # 读取原始音频（44100Hz）
            wave, sr = torchaudio.load(wav_path)
            
            # 严格重采样到 16000Hz
            if sr != TARGET_SAMPLE_RATE:
                wave = torchaudio.functional.resample(wave, sr, TARGET_SAMPLE_RATE)
                
            if wave.ndim == 2:
                wave = wave.mean(0, keepdim=True)
            
            total_samples = wave.shape[1]
            
            # --- 核心切片逻辑 ---
            for i in range(MAX_SEGMENTS):
                start_sample = i * SEGMENT_SAMPLES
                end_sample = start_sample + SEGMENT_SAMPLES
                
                # 如果起始点已经超过音频总长，停止切片
                if start_sample >= total_samples:
                    break
                    
                # 截取10秒片段
                chunk = wave[:, start_sample:end_sample]
                
                # 如果最后一段因微弱误差不足10秒，进行后补零(Padding)
                if chunk.shape[1] < SEGMENT_SAMPLES:
                    pad_len = SEGMENT_SAMPLES - chunk.shape[1]
                    chunk = torch.cat([chunk, torch.zeros((1, pad_len))], dim=1)
                
                # 对当前10秒片段进行模型预测
                with torch.no_grad():
                    logits = model(chunk.to(DEVICE)).squeeze(0)  # 输出 Raw Logits
                    # 独立的 10 秒片段直接通过 argmax 得到单段的预报结果
                    pred_label = torch.argmax(logits, dim=-1).item()
                
                # 【修改核心】不再进行多段聚合，每一个 10 秒片段都视作独立样本追加入评估集
                y_true.append(true_label)
                y_pred.append(pred_label)
                
        except Exception as e:
            print(f"读取或处理文件失败 {wav_name}: {e}")

# ==================== 3. 统计各项核心指标 ====================
y_true = np.array(y_true)
y_pred = np.array(y_pred)

accuracy = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

print("\n" + "="*20 + " 10秒独立片段综合评估结果 " + "="*20)
print(f"总计成功评估 10秒片段数 : {len(y_true)}")
print(f"分段预报准确率 (Segment Accuracy): {accuracy * 100:.2f}%")
print("-" * 50)

print("各声音类别的独立【误报率 (FPR)】统计:")
for i in range(4):
    fp = np.sum((y_pred == i) & (y_true != i))
    tn = np.sum((y_pred != i) & (y_true != i))
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    print(f"  * [{LABEL_MAP[i]}] 的误报率: {fpr * 100:.2f}%")

print("-" * 50)
print("文本级混淆矩阵 (横行代表真实类别，纵列代表模型预测结果):")
print(f"{'真实   预测':<15}{'鸟声':<10}{'施工作业':<10}{'雷声':<10}{'其他(Noise)':<10}")
for i in range(4):
    row_str = f"{LABEL_MAP[i]:<14}"
    for j in range(4):
        row_str += f"{cm[i, j]:<12}"
    print(row_str)

# ==================== 4. 生成可视化图形矩阵 ====================
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(8, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei']  
    plt.rcParams['axes.unicode_minus'] = False
    
    labels_text = [LABEL_MAP[i] for i in range(4)]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels_text, yticklabels=labels_text)
    plt.title('10秒片段独立分类混淆矩阵')
    plt.ylabel('真实文件夹标签')
    plt.xlabel('模型预测标签')
    
    output_img = os.path.join(os.path.dirname(CHECKPOINT_PATH), "confusion_matrix_split.png")
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"\n可视化混淆矩阵已成功保存至: {output_img}")
except ImportError:
    print("\n提示: 未安装 matplotlib 或 seaborn，已跳过图片显示。")