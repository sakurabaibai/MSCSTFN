import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
from PIL import Image
import os
from vmdpy import VMD  # 导入VMD库

# 创建保存图像的文件夹
os.makedirs('IMF1_images', exist_ok=True)
os.makedirs('IMF2_images', exist_ok=True)

# 读取整个数据文件
data = pd.read_csv('fusion_results/fused_signal.csv', sep='\t', header=None)
full_signal = data.iloc[:, 0].to_numpy()

# 计算总段数 (每段1024个点)
num_segments = len(full_signal) // 1024
# 确保不超过100段
num_segments = min(num_segments, 100)

# VMD参数设置
alpha = 2000  # 带宽约束参数
tau = 0  # 噪声容忍度
K = 2  # 分解模态数量（IMF个数）
DC = 0  # 不包含直流分量
init = 1  # 初始化中心频率为均匀分布
tol = 1e-7  # 收敛容差

for seg_idx in range(num_segments):
    print(f"Processing segment {seg_idx + 1}/{num_segments}")

    # 提取当前段的数据 (1024个点)
    start_idx = seg_idx * 1024
    end_idx = start_idx + 1024
    X = full_signal[start_idx:end_idx]
    t = np.linspace(0, 1, 1024)

    # VMD分解 - 替换CEEMDAN
    u, u_hat, omega = VMD(X, alpha, tau, K, DC, init, tol)

    # 确保分解得到至少两个IMF
    if u.shape[0] < 2:
        print(f"Warning: Segment {seg_idx + 1} produced less than 2 IMFs")
        continue

    # 处理前两个IMF
    for i in range(2):
        imf = u[i, :]
        # 连续小波变换
        coefs, freqs = pywt.cwt(imf, np.arange(1, 50), 'mexh')
        magnitude = abs(coefs)

        # 归一化并应用彩色映射
        norm_magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())
        colored_image = plt.cm.jet(norm_magnitude)

        # 转换为PIL图像并调整大小
        pil_image = Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8))
        resized_image = pil_image.resize((224, 224))

        # 保存图像
        if i == 0:
            resized_image.save(f'IMF1_images/{seg_idx + 1}.png')
        else:
            resized_image.save(f'IMF2_images/{seg_idx + 1}.png')

print("Processing completed. All images saved.")