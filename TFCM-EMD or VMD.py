import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyEMD import CEEMDAN, EMD
import pywt
from vmdpy import VMD
from scipy import stats
from PIL import Image
import os
import time  # 导入时间模块用于耗时计算

# 创建保存图像的文件夹
os.makedirs('IMF1_images', exist_ok=True)
os.makedirs('IMF2_images', exist_ok=True)

# 初始化结果存储
ceemdan_results = {
    'modo_mixing': [],
    'boundary_effect': [],
    'time': []  # 添加CEEMDAN耗时记录
}

vmd_results = {
    'modo_mixing': [],
    'boundary_effect': [],
    'time': []  # 添加VMD耗时记录
}

# 读取整个数据文件
data = pd.read_csv('fusion_results/fused_signal.csv', sep='\t', header=None)
full_signal = data.iloc[:, 0].to_numpy()

# 计算总段数 (每段1024个点)
num_segments = len(full_signal) // 1024
# 确保不超过10段
num_segments = min(num_segments, 10)

ceemdan = CEEMDAN()
emd = EMD()  # 用于计算模态混叠

# VMD参数设置
alpha = 2000  # 带宽约束
tau = 0.  # 噪声容忍度
K = 5  # 模态数量
DC = 0  # 无直流分量
init = 1  # 初始化omega为均匀分布
tol = 1e-7  # 收敛容差

for seg_idx in range(num_segments):
    print(f"Processing segment {seg_idx + 1}/{num_segments}")

    # 提取当前段的数据 (1024个点)
    start_idx = seg_idx * 1024
    end_idx = start_idx + 1024
    X = full_signal[start_idx:end_idx]
    t = np.linspace(0, 1, 1024)

    # ================== CEEMDAN 分解 ==================
    start_time = time.time()  # 记录开始时间

    ceemdan_imfs = ceemdan.ceemdan(X)

    ceemdan_time = time.time() - start_time  # 计算耗时
    ceemdan_results['time'].append(ceemdan_time)

    # 确保至少有两个IMF
    if len(ceemdan_imfs) < 2:
        print(f"Warning: CEEMDAN Segment {seg_idx + 1} produced less than 2 IMFs")
    else:
        # 计算模态混叠指标 (样本熵)
        ceemdan_entropy = []
        for imf in ceemdan_imfs[:2]:
            # 计算样本熵
            r = 0.2 * np.std(imf)
            n = len(imf)
            matches = 0
            total = 0

            for i in range(n - 1):
                for j in range(i + 1, n - 1):
                    if abs(imf[i] - imf[j]) < r and abs(imf[i + 1] - imf[j + 1]) < r:
                        matches += 1
                    total += 1

            ceemdan_entropy.append(-np.log(matches / total) if matches > 0 else 0)

        # 计算边界效应指标 (端点能量比)
        boundary_energy = np.sum(ceemdan_imfs[0][:50] ** 2) + np.sum(ceemdan_imfs[0][-50:] ** 2)
        total_energy = np.sum(ceemdan_imfs[0] ** 2)
        boundary_ratio = boundary_energy / total_energy

        ceemdan_results['modo_mixing'].append(np.mean(ceemdan_entropy))
        ceemdan_results['boundary_effect'].append(boundary_ratio)

        # 可视化处理 (仅IMF1和IMF2)
        for i in range(min(2, len(ceemdan_imfs))):
            imf = ceemdan_imfs[i]
            # 连续小波变换
            coefs, freqs = pywt.cwt(imf, np.arange(1, 50), 'mexh')
            magnitude = abs(coefs)

            # 归一化并应用彩色映射
            norm_magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())
            colored_image = plt.cm.jet(norm_magnitude)

            # 转换为 PIL 图像并调整大小
            pil_image = Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8))
            resized_image = pil_image.resize((224, 224))

            # 保存图像
            if i == 0:
                resized_image.save(f'IMF1_images/ceemdan_{seg_idx + 1}.png')
            else:
                resized_image.save(f'IMF2_images/ceemdan_{seg_idx + 1}.png')

    # ================== VMD 分解 ==================
    start_time = time.time()  # 记录开始时间

    # 执行VMD分解
    vmd_imfs, _, _ = VMD(X, alpha, tau, K, DC, init, tol)

    vmd_time = time.time() - start_time  # 计算耗时
    vmd_results['time'].append(vmd_time)

    if vmd_imfs.shape[0] < 2:
        print(f"Warning: VMD Segment {seg_idx + 1} produced less than 2 IMFs")
    else:
        # 计算模态混叠指标 (样本熵)
        vmd_entropy = []
        for imf in vmd_imfs[:2]:
            # 计算样本熵
            r = 0.2 * np.std(imf)
            n = len(imf)
            matches = 0
            total = 0

            for i in range(n - 1):
                for j in range(i + 1, n - 1):
                    if abs(imf[i] - imf[j]) < r and abs(imf[i + 1] - imf[j + 1]) < r:
                        matches += 1
                    total += 1

            vmd_entropy.append(-np.log(matches / total) if matches > 0 else 0)

        # 计算边界效应指标 (端点能量比)
        boundary_energy = np.sum(vmd_imfs[0][:50] ** 2) + np.sum(vmd_imfs[0][-50:] ** 2)
        total_energy = np.sum(vmd_imfs[0] ** 2)
        boundary_ratio = boundary_energy / total_energy

        vmd_results['modo_mixing'].append(np.mean(vmd_entropy))
        vmd_results['boundary_effect'].append(boundary_ratio)

        # 可视化处理 (仅IMF1和IMF2)
        for i in range(min(2, len(vmd_imfs))):
            imf = vmd_imfs[i]
            # 连续小波变换
            coefs, freqs = pywt.cwt(imf, np.arange(1, 50), 'mexh')
            magnitude = abs(coefs)

            # 归一化并应用彩色映射
            norm_magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())
            colored_image = plt.cm.jet(norm_magnitude)

            # 转换为 PIL 图像并调整大小
            pil_image = Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8))
            resized_image = pil_image.resize((224, 224))

            # 保存图像
            if i == 0:
                resized_image.save(f'IMF1_images/vmd_{seg_idx + 1}.png')
            else:
                resized_image.save(f'IMF2_images/vmd_{seg_idx + 1}.png')

# ================== 结果对比 ==================
print("\n===== 性能对比 =====")
# 模态混叠对比 (样本熵越高混叠越严重)
ceemdan_mm = np.mean(ceemdan_results['modo_mixing'])
vmd_mm = np.mean(vmd_results['modo_mixing'])
print(f"平均模态混叠概率: CEEMDAN={ceemdan_mm:.4f}, VMD={vmd_mm:.4f}")

# 边界效应对比 (比值越高边界效应越严重)
ceemdan_be = np.mean(ceemdan_results['boundary_effect'])
vmd_be = np.mean(vmd_results['boundary_effect'])
print(f"平均边界效应指数: CEEMDAN={ceemdan_be:.4f}, VMD={vmd_be:.4f}")

# 统计显著性检验
t_stat, p_value = stats.ttest_ind(
    ceemdan_results['modo_mixing'],
    vmd_results['modo_mixing'],
    equal_var=False
)
print(f"模态混叠显著性(p值): {p_value:.6f} {'(显著差异)' if p_value < 0.05 else '(无显著差异)'}")

t_stat, p_value = stats.ttest_ind(
    ceemdan_results['boundary_effect'],
    vmd_results['boundary_effect'],
    equal_var=False
)
print(f"边界效应显著性(p值): {p_value:.6f} {'(显著差异)' if p_value < 0.05 else '(无显著差异)'}")

# 耗时对比
ceemdan_avg_time = np.mean(ceemdan_results['time'])
vmd_avg_time = np.mean(vmd_results['time'])
ceemdan_total_time = np.sum(ceemdan_results['time'])
vmd_total_time = np.sum(vmd_results['time'])

print("\n===== 耗时对比 =====")
print(f"平均耗时(秒/段): CEEMDAN={ceemdan_avg_time:.4f}, VMD={vmd_avg_time:.4f}")
print(f"总耗时(秒): CEEMDAN={ceemdan_total_time:.2f}, VMD={vmd_total_time:.2f}")
print(f"速度比率: VMD是CEEMDAN的 {ceemdan_avg_time / vmd_avg_time:.2f} 倍快")

# # 生成对比图表
# plt.figure(figsize=(15, 10))
#
# # 模态混叠对比
# plt.subplot(231)
# plt.boxplot([ceemdan_results['modo_mixing'], vmd_results['modo_mixing']],
#             labels=['CEEMDAN', 'VMD'])
# plt.title('模态混叠概率对比 (样本熵)')
# plt.ylabel('样本熵值')
#
# # 边界效应对比
# plt.subplot(232)
# plt.boxplot([ceemdan_results['boundary_effect'], vmd_results['boundary_effect']],
#             labels=['CEEMDAN', 'VMD'])
# plt.title('边界效应指数对比')
# plt.ylabel('边界能量比')
#
# # 耗时对比
# plt.subplot(233)
# plt.bar(['CEEMDAN', 'VMD'], [ceemdan_avg_time, vmd_avg_time], color=['blue', 'orange'])
# plt.title('平均分解耗时对比')
# plt.ylabel('时间 (秒)')
# plt.text(0, ceemdan_avg_time, f'{ceemdan_avg_time:.4f}', ha='center', va='bottom')
# plt.text(1, vmd_avg_time, f'{vmd_avg_time:.4f}', ha='center', va='bottom')
#
# # 耗时分布箱线图
# plt.subplot(234)
# plt.boxplot([ceemdan_results['time'], vmd_results['time']],
#             labels=['CEEMDAN', 'VMD'])
# plt.title('分解耗时分布')
# plt.ylabel('时间 (秒)')
# plt.yscale('log')  # 使用对数刻度更好显示差异
#
# # 耗时分布直方图
# plt.subplot(235)
# plt.hist(ceemdan_results['time'], bins=20, alpha=0.5, label='CEEMDAN')
# plt.hist(vmd_results['time'], bins=20, alpha=0.5, label='VMD')
# plt.title('耗时分布直方图')
# plt.xlabel('时间 (秒)')
# plt.ylabel('频数')
# plt.legend()
#
# # 耗时与信号复杂度关系
# plt.subplot(236)
# plt.scatter(ceemdan_results['modo_mixing'], ceemdan_results['time'], alpha=0.5, label='CEEMDAN')
# plt.scatter(vmd_results['modo_mixing'], vmd_results['time'], alpha=0.5, label='VMD')
# plt.title('模态混叠与耗时关系')
# plt.xlabel('模态混叠指数 (样本熵)')
# plt.ylabel('分解耗时 (秒)')
# plt.legend()
#
# plt.tight_layout()
# plt.savefig('comparison_results.png')
# plt.show()
#
# print("处理完成，结果已保存!")