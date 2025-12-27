import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
from matplotlib.lines import Line2D
from einops import rearrange, repeat

from models.TSViT.TSViTdense import TSViT

# 设置保存路径
save_dir = './gradient_visualization'
os.makedirs(save_dir, exist_ok=True)

# 加载配置文件
config_path = r'C:\Users\Think\Desktop\DeepSatModels-main\configs\PASTIS24/TSViT_fold5.yaml'
  # 替换为你的实际路径
with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)
model_cfg = cfg['MODEL']

# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TSViT(model_cfg).to(device)

# 加载预训练模型
checkpoint_path = r'C:\Users\Think\Desktop\模型\logs\门控自适应8684\best.pth'
  # 替换为你的权重文件路径
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint)
model.eval()

# 准备输入数据
B = 1
T = model_cfg['max_seq_len']
C = model_cfg['num_channels']
H = W = model_cfg['img_res']
x = torch.randn(B, T, C, H, W).to(device)
x.requires_grad = True

# 前向传播 + 最大预测概率
logits = model(x)  # [B, num_classes, H, W]
probs = torch.softmax(logits, dim=1)
max_prob = probs.max()
max_prob.backward()

# 获取输入梯度
grad = x.grad[0].detach().cpu().numpy()   # [T, C, H, W]
x_np = x.detach().cpu().numpy()[0]        # [T, C, H, W]

# 可视化：平均空间维度后绘制每个通道的变化
fig, axs = plt.subplots(2, figsize=(12, 10))

mean_input = x_np.mean(axis=(2, 3))   # [T, C]
mean_grad = grad.mean(axis=(2, 3))    # [T, C]

axs[0].plot(mean_input)
axs[0].set_title("Input Mean Over Spatial Dimensions")
axs[0].set_ylabel("x")

axs[1].plot(mean_grad, linewidth=2)
axs[1].set_title("Gradient Mean Over Spatial Dimensions (dy/dx)")
axs[1].set_ylabel("dy/dx")

plt.savefig(os.path.join(save_dir, "input_and_gradient_plot.png"))
plt.close()

print(f"Gradient visualization saved to {save_dir}")
