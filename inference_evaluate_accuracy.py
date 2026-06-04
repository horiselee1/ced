import os
import torch
import torchaudio
import numpy as np
import models
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix

# ==================== 配置区 ====================
# 1. 四个测试文件夹所在的根目录
TEST_ROOT_DIR = r"D:\vscode\ceshi\test800"  

# 2. 模型权重路径
CHECKPOINT_PATH = "./experiments/my_audiotransformer_mini/audiotransformer_mini/2026-06-03_18-31_561c97535f3711f18621d8459082dc2d/best_checkpoint_19_0.9947.pt"

# 3. 文件夹名到模型标签索引的映射（严格对应 LABEL_MAP）
FOLDER_TO_IDX = {
    "Bird": 0,
    "Construction": 1,
    "Thunder": 2,
    "Noise": 3
}

LABEL_MAP = {0: "鸟声", 1: "施工作业声", 2: "雷声", 3: "其他声音(Noise)"}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== 1. 初始化并加载模型 ====================
print("正在加载模型与权重...")
model = models.audiotransformer_mini(num_classes=4, pretrained=False)
ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
model.load_state_dict(ckpt['model'])
model = model.to(DEVICE).eval()

# ==================== 2. 批量扫描音频与推理 ====================
y_true = []
y_pred = []

print("开始批量扫描音频并进行推理...")
for folder_name, true_label in FOLDER_TO_IDX.items():
    folder_path = os.path.join(TEST_ROOT_DIR, folder_name)
    
    if not os.path.exists(folder_path):
        print(f"警告: 文件夹不存在，跳过 -> {folder_path}")
        continue
        
    # 获取文件夹下所有的 wav 文件
    wav_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.wav')]
    print(f"正在处理文件夹 [{folder_name}]，共发现 {len(wav_files)} 个音频...")
    
    for wav_name in tqdm(wav_files, desc=f"推理 {folder_name}"):
        wav_path = os.path.join(folder_path, wav_name)
        
        try:
            # 读取音频
            wave, sr = torchaudio.load(wav_path)
            if sr != 16000:
                wave = torchaudio.functional.resample(wave, sr, 16000)
                
            if wave.ndim == 2:
                wave = wave.mean(0, keepdim=True)
                
            # 模型推理
            with torch.no_grad():
                output = model(wave.to(DEVICE)).squeeze(0)
                pred_label = torch.argmax(output).item()
                
            # 记录真实标签与预测标签
            y_true.append(true_label)
            y_pred.append(pred_label)
            
        except Exception as e:
            print(f"读取或处理文件失败 {wav_name}: {e}")

# ==================== 3. 统计各项核心指标 ====================
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# 计算整体准确率
accuracy = accuracy_score(y_true, y_pred)

# 计算标准混淆矩阵 (行代表真实标签，列代表预测标签)
cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

print("\n" + "="*20 + " 评估统计结果 " + "="*20)
print(f"总计成功评估样本数: {len(y_true)}")
print(f"模型全品类预报准确率 (Accuracy): {accuracy * 100:.2f}%")
print("-" * 50)

# 计算每个类别的独立预报误报率 (False Positive Rate)
# 误报率公式 = 某类别的(把其他声音错误预测为本声音的数量) / (实际不是本声音的总数量)
print("各声音类别的独立【误报率 (FPR)】统计:")
for i in range(4):
    fp = np.sum((y_pred == i) & (y_true != i))  # 预测是i但实际不是i
    tn = np.sum((y_pred != i) & (y_true != i))  # 预测不是i实际也不是i
    
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

# ==================== 4. 可选：生成可视化图形矩阵 ====================
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(8, 6))
    # 解决 matplotlib 中文乱码问题
    plt.rcParams['font.sans-serif'] = ['SimHei']  
    plt.rcParams['axes.unicode_minus'] = False
    
    labels_text = [LABEL_MAP[i] for i in range(4)]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels_text, yticklabels=labels_text)
    plt.title('音频声音分类混淆矩阵')
    plt.ylabel('真实文件夹标签')
    plt.xlabel('模型预测标签')
    
    output_img = os.path.join(os.path.dirname(CHECKPOINT_PATH), "confusion_matrix.png")
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"\n可视化混淆矩阵已成功保存至: {output_img}")
except ImportError:
    print("\n提示: 未安装 matplotlib 或 seaborn，已跳过图片矩阵生成，上面的文本矩阵同样精准。")