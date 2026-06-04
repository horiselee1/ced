import torch
import torchaudio
import models

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 建立你的 4 类别中文映射映射
LABEL_MAP = {0: "鸟声", 1: "施工作业声", 2: "雷声", 3: "其他声音"}

# 1. 初始化 4 分类的 mini 模型
model = models.audiotransformer_mini(num_classes=4, pretrained=False)

# 2. 加载你微调训练出来的最佳 checkpoint 权重 (注意替换你实际的路径)
checkpoint_path = "./experiments/my_audiotransformer_mini/audiotransformer_mini/2026-06-03_18-31_561c97535f3711f18621d8459082dc2d/best_checkpoint_17_0.9943.pt"
ckpt = torch.load(checkpoint_path, map_location="cpu")
model.load_state_dict(ckpt['model'])
model = model.to(DEVICE).eval()

# 3. 读取待测试的波形
wav_path = r"D:\vscode\ceshi\test800\Noise\Noise3- (87).wav"
wave, sr = torchaudio.load(wav_path)
if sr != 16000:
    wave = torchaudio.functional.resample(wave, sr, 16000)

# 4. 推理并打印概率
with torch.no_grad():
    # 转换为模型接受的输入形状 (batch, samples)
    if wave.ndim == 2:
        wave = wave.mean(0, keepdim=True) # 降维单声道
        
    output = model(wave.to(DEVICE)).squeeze(0) # 得到 4 维的概率张量
    
    print(f"===== 音频预测结果 =====")
    # 打印前 2 个置信度最高的类别
    probs, indices = output.topk(1)
    for prob, idx in zip(probs, indices):
        print(f"检测到事件: {LABEL_MAP[idx.item()]:<10} | 置信度(概率): {prob.item():.4f}")