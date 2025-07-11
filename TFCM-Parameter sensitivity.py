import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyEMD import CEEMDAN, EMD
import pywt
from vmdpy import VMD
from scipy import stats
from PIL import Image
import os
import time
import itertools  # 用于参数组合

# 创建保存图像的文件夹
os.makedirs('IMF1_images', exist_ok=True)
os.makedirs('IMF2_images', exist_ok=True)
os.makedirs('sensitivity_results', exist_ok=True)

# 初始化结果存储
ceemdan_results = {
    'modo_mixing': [],
    'boundary_effect': [],
    'time': []
}

vmd_results = {
    'modo_mixing': [],
    'boundary_effect': [],
    'time': []
}

# 敏感性分析结果存储
sensitivity_results = {
    'ceemdan': {
        'noise_amplitude': [],
        'decomposition_level': [],
        'wavelet_type': [],
        'modo_mixing': [],
        'boundary_effect': [],
        'time': []
    },
    'vmd': {
        'noise_amplitude': [],
        'decomposition_level': [],  # 对于VMD就是K值
        'wavelet_type': [],
        'modo_mixing': [],
        'boundary_effect': [],
        'time': []
    }
}

# 读取整个数据文件
data = pd.read_csv('fusion_results/fused_signal.csv', sep='\t', header=None)
full_signal = data.iloc[:, 0].to_numpy()

# 计算总段数 (每段1024个点)
num_segments = len(full_signal) // 1024
# 确保不超过10段
num_segments = min(num_segments, 10)

# VMD参数设置
alpha = 2000  # 带宽约束
tau = 0.  # 噪声容忍度
DC = 0  # 无直流分量
init = 1  # 初始化omega为均匀分布
tol = 1e-7  # 收敛容差

# 敏感性参数范围
noise_amplitudes = [0.0, 0.05, 0.1, 0.2]  # 噪声幅度 (标准差比例)
ceemdan_levels = [5, 8, 12]  # CEEMDAN分解层数
vmd_levels = [3, 5, 7]  # VMD分解层数 (K值)
wavelet_types = ['mexh', 'morl', 'cgau8']  # 小波类型


def calculate_sample_entropy(imf):
    """计算样本熵"""
    r = 0.2 * np.std(imf)
    n = len(imf)
    matches = 0
    total = 0

    for i in range(n - 1):
        for j in range(i + 1, n - 1):
            if abs(imf[i] - imf[j]) < r and abs(imf[i + 1] - imf[j + 1]) < r:
                matches += 1
            total += 1

    return -np.log(matches / total) if matches > 0 else 0


def calculate_boundary_effect(imf):
    """计算边界效应指标"""
    boundary_energy = np.sum(imf[:50] ** 2) + np.sum(imf[-50:] ** 2)
    total_energy = np.sum(imf ** 2)
    return boundary_energy / total_energy


def process_segment(seg_idx, X, wavelet_type, noise_amp=0.0, ceemdan_level=8, vmd_level=5):
    """处理单个信号段"""
    t = np.linspace(0, 1, 1024)

    # 添加噪声
    if noise_amp > 0:
        noise = np.random.normal(0, noise_amp * np.std(X), len(X))
        X = X + noise

    # ================== CEEMDAN 分解 ==================
    start_time = time.time()
    ceemdan = CEEMDAN(max_imf=ceemdan_level)
    ceemdan_imfs = ceemdan.ceemdan(X)
    ceemdan_time = time.time() - start_time

    # 确保至少有两个IMF
    if len(ceemdan_imfs) < 2:
        print(f"Warning: CEEMDAN Segment {seg_idx} produced less than 2 IMFs")
        ceemdan_mm = np.nan
        ceemdan_be = np.nan
    else:
        # 计算模态混叠指标 (样本熵)
        ceemdan_entropy = [calculate_sample_entropy(imf) for imf in ceemdan_imfs[:2]]
        ceemdan_mm = np.mean(ceemdan_entropy)

        # 计算边界效应指标 (端点能量比)
        ceemdan_be = calculate_boundary_effect(ceemdan_imfs[0])

        # 可视化处理 (仅IMF1和IMF2)
        for i in range(min(2, len(ceemdan_imfs))):
            imf = ceemdan_imfs[i]
            # 连续小波变换
            coefs, freqs = pywt.cwt(imf, np.arange(1, 50), wavelet_type)
            magnitude = abs(coefs)

            # 归一化并应用彩色映射
            norm_magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())
            colored_image = plt.cm.jet(norm_magnitude)

            # 转换为 PIL 图像并调整大小
            pil_image = Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8))
            resized_image = pil_image.resize((224, 224))

            # 保存图像
            img_dir = 'IMF1_images' if i == 0 else 'IMF2_images'
            img_name = f'ceemdan_s{seg_idx}_n{noise_amp:.2f}_l{ceemdan_level}_w{wavelet_type}.png'
            resized_image.save(f'{img_dir}/{img_name}')

    # ================== VMD 分解 ==================
    start_time = time.time()
    vmd_imfs, _, _ = VMD(X, alpha, tau, vmd_level, DC, init, tol)
    vmd_time = time.time() - start_time

    if vmd_imfs.shape[0] < 2:
        print(f"Warning: VMD Segment {seg_idx} produced less than 2 IMFs")
        vmd_mm = np.nan
        vmd_be = np.nan
    else:
        # 计算模态混叠指标 (样本熵)
        vmd_entropy = [calculate_sample_entropy(imf) for imf in vmd_imfs[:2]]
        vmd_mm = np.mean(vmd_entropy)

        # 计算边界效应指标 (端点能量比)
        vmd_be = calculate_boundary_effect(vmd_imfs[0])

        # 可视化处理 (仅IMF1和IMF2)
        for i in range(min(2, len(vmd_imfs))):
            imf = vmd_imfs[i]
            # 连续小波变换
            coefs, freqs = pywt.cwt(imf, np.arange(1, 50), wavelet_type)
            magnitude = abs(coefs)

            # 归一化并应用彩色映射
            norm_magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())
            colored_image = plt.cm.jet(norm_magnitude)

            # 转换为 PIL 图像并调整大小
            pil_image = Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8))
            resized_image = pil_image.resize((224, 224))

            # 保存图像
            img_dir = 'IMF1_images' if i == 0 else 'IMF2_images'
            img_name = f'vmd_s{seg_idx}_n{noise_amp:.2f}_k{vmd_level}_w{wavelet_type}.png'
            resized_image.save(f'{img_dir}/{img_name}')

    return (ceemdan_mm, ceemdan_be, ceemdan_time), (vmd_mm, vmd_be, vmd_time)


# 主处理循环
for seg_idx in range(num_segments):
    print(f"\nProcessing segment {seg_idx + 1}/{num_segments}")

    # 提取当前段的数据 (1024个点)
    start_idx = seg_idx * 1024
    end_idx = start_idx + 1024
    X_original = full_signal[start_idx:end_idx]

    # 首先进行基准测试（无噪声，默认参数）
    (ceemdan_mm, ceemdan_be, ceemdan_time), (vmd_mm, vmd_be, vmd_time) = process_segment(
        seg_idx + 1, X_original, 'mexh'
    )

    if not np.isnan(ceemdan_mm):
        ceemdan_results['modo_mixing'].append(ceemdan_mm)
        ceemdan_results['boundary_effect'].append(ceemdan_be)
        ceemdan_results['time'].append(ceemdan_time)

    if not np.isnan(vmd_mm):
        vmd_results['modo_mixing'].append(vmd_mm)
        vmd_results['boundary_effect'].append(vmd_be)
        vmd_results['time'].append(vmd_time)

    # 只对第一段进行全面的敏感性分析
    if seg_idx == 0:
        print("Performing sensitivity analysis on segment 1...")

        # 生成所有参数组合
        param_combinations = list(itertools.product(
            noise_amplitudes,
            ceemdan_levels,
            vmd_levels,
            wavelet_types
        ))

        total_combinations = len(param_combinations)
        print(f"Total parameter combinations: {total_combinations}")

        for i, (noise_amp, ceemdan_level, vmd_level, wavelet) in enumerate(param_combinations):
            print(f"\nProcessing combination {i + 1}/{total_combinations}: "
                  f"noise={noise_amp:.2f}, CEEMDAN_level={ceemdan_level}, "
                  f"VMD_level={vmd_level}, wavelet={wavelet}")

            # 处理当前参数组合
            (ceemdan_mm, ceemdan_be, ceemdan_time), (vmd_mm, vmd_be, vmd_time) = process_segment(
                1, X_original, wavelet, noise_amp, ceemdan_level, vmd_level
            )

            # 存储CEEMDAN敏感性结果
            sensitivity_results['ceemdan']['noise_amplitude'].append(noise_amp)
            sensitivity_results['ceemdan']['decomposition_level'].append(ceemdan_level)
            sensitivity_results['ceemdan']['wavelet_type'].append(wavelet)
            sensitivity_results['ceemdan']['modo_mixing'].append(ceemdan_mm)
            sensitivity_results['ceemdan']['boundary_effect'].append(ceemdan_be)
            sensitivity_results['ceemdan']['time'].append(ceemdan_time)

            # 存储VMD敏感性结果
            sensitivity_results['vmd']['noise_amplitude'].append(noise_amp)
            sensitivity_results['vmd']['decomposition_level'].append(vmd_level)
            sensitivity_results['vmd']['wavelet_type'].append(wavelet)
            sensitivity_results['vmd']['modo_mixing'].append(vmd_mm)
            sensitivity_results['vmd']['boundary_effect'].append(vmd_be)
            sensitivity_results['vmd']['time'].append(vmd_time)

# 保存敏感性结果
ceemdan_sens_df = pd.DataFrame({
    'noise_amplitude': sensitivity_results['ceemdan']['noise_amplitude'],
    'decomposition_level': sensitivity_results['ceemdan']['decomposition_level'],
    'wavelet_type': sensitivity_results['ceemdan']['wavelet_type'],
    'modo_mixing': sensitivity_results['ceemdan']['modo_mixing'],
    'boundary_effect': sensitivity_results['ceemdan']['boundary_effect'],
    'time': sensitivity_results['ceemdan']['time']
})
ceemdan_sens_df.to_csv('sensitivity_results/ceemdan_sensitivity.csv', index=False)

vmd_sens_df = pd.DataFrame({
    'noise_amplitude': sensitivity_results['vmd']['noise_amplitude'],
    'decomposition_level': sensitivity_results['vmd']['decomposition_level'],
    'wavelet_type': sensitivity_results['vmd']['wavelet_type'],
    'modo_mixing': sensitivity_results['vmd']['modo_mixing'],
    'boundary_effect': sensitivity_results['vmd']['boundary_effect'],
    'time': sensitivity_results['vmd']['time']
})
vmd_sens_df.to_csv('sensitivity_results/vmd_sensitivity.csv', index=False)

# ================== 结果对比 ==================
print("\n===== 性能对比 =====")
# 模态混叠对比 (样本熵越高混叠越严重)
ceemdan_mm = np.nanmean(ceemdan_results['modo_mixing'])
vmd_mm = np.nanmean(vmd_results['modo_mixing'])
print(f"平均模态混叠概率: CEEMDAN={ceemdan_mm:.4f}, VMD={vmd_mm:.4f}")

# 边界效应对比 (比值越高边界效应越严重)
ceemdan_be = np.nanmean(ceemdan_results['boundary_effect'])
vmd_be = np.nanmean(vmd_results['boundary_effect'])
print(f"平均边界效应指数: CEEMDAN={ceemdan_be:.4f}, VMD={vmd_be:.4f}")

# 统计显著性检验
t_stat, p_value = stats.ttest_ind(
    ceemdan_results['modo_mixing'],
    vmd_results['modo_mixing'],
    nan_policy='omit'
)
print(f"模态混叠显著性(p值): {p_value:.6f} {'(显著差异)' if p_value < 0.05 else '(无显著差异)'}")

t_stat, p_value = stats.ttest_ind(
    ceemdan_results['boundary_effect'],
    vmd_results['boundary_effect'],
    nan_policy='omit'
)
print(f"边界效应显著性(p值): {p_value:.6f} {'(显著差异)' if p_value < 0.05 else '(无显著差异)'}")

# 耗时对比
ceemdan_avg_time = np.nanmean(ceemdan_results['time'])
vmd_avg_time = np.nanmean(vmd_results['time'])
ceemdan_total_time = np.nansum(ceemdan_results['time'])
vmd_total_time = np.nansum(vmd_results['time'])

print("\n===== 耗时对比 =====")
print(f"平均耗时(秒/段): CEEMDAN={ceemdan_avg_time:.4f}, VMD={vmd_avg_time:.4f}")
print(f"总耗时(秒): CEEMDAN={ceemdan_total_time:.2f}, VMD={vmd_total_time:.2f}")
print(f"速度比率: VMD是CEEMDAN的 {ceemdan_avg_time / vmd_avg_time:.2f} 倍快")

# ================== 敏感性分析总结 ==================
print("\n===== 敏感性分析总结 =====")
print("CEEMDAN敏感性:")
print(f"噪声幅度影响: {ceemdan_sens_df.groupby('noise_amplitude')['modo_mixing'].mean()}")
print(f"分解层数影响: {ceemdan_sens_df.groupby('decomposition_level')['modo_mixing'].mean()}")
print(f"小波类型影响: {ceemdan_sens_df.groupby('wavelet_type')['modo_mixing'].mean()}")

print("\nVMD敏感性:")
print(f"噪声幅度影响: {vmd_sens_df.groupby('noise_amplitude')['modo_mixing'].mean()}")
print(f"分解层数影响: {vmd_sens_df.groupby('decomposition_level')['modo_mixing'].mean()}")
print(f"小波类型影响: {vmd_sens_df.groupby('wavelet_type')['modo_mixing'].mean()}")

# 绘制敏感性分析结果
plt.figure(figsize=(15, 10))

# CEEMDAN噪声敏感性
plt.subplot(2, 3, 1)
for level in ceemdan_levels:
    subset = ceemdan_sens_df[ceemdan_sens_df['decomposition_level'] == level]
    plt.plot(subset['noise_amplitude'], subset['modo_mixing'], 'o-', label=f'N={level}')
plt.xlabel('Noise Amplitude')
plt.ylabel('Modo Mixing')
plt.title('CEEMDAN: Noise Sensitivity')
plt.legend()

# CEEMDAN分解层数敏感性
plt.subplot(2, 3, 2)
for noise in noise_amplitudes:
    subset = ceemdan_sens_df[ceemdan_sens_df['noise_amplitude'] == noise]
    plt.plot(subset['decomposition_level'], subset['modo_mixing'], 'o-', label=f'Noise={noise}')
plt.xlabel('Decomposition Level')
plt.ylabel('Modo Mixing')
plt.title('CEEMDAN: Level Sensitivity')
plt.legend()

# CEEMDAN小波类型敏感性
plt.subplot(2, 3, 3)
ceemdan_sens_df.boxplot(column='modo_mixing', by='wavelet_type', ax=plt.gca())
plt.xlabel('Wavelet Type')
plt.ylabel('Modo Mixing')
plt.title('CEEMDAN: Wavelet Sensitivity')
plt.suptitle('')

# VMD噪声敏感性
plt.subplot(2, 3, 4)
for level in vmd_levels:
    subset = vmd_sens_df[vmd_sens_df['decomposition_level'] == level]
    plt.plot(subset['noise_amplitude'], subset['modo_mixing'], 'o-', label=f'K={level}')
plt.xlabel('Noise Amplitude')
plt.ylabel('Modo Mixing')
plt.title('VMD: Noise Sensitivity')
plt.legend()

# VMD分解层数敏感性
plt.subplot(2, 3, 5)
for noise in noise_amplitudes:
    subset = vmd_sens_df[vmd_sens_df['noise_amplitude'] == noise]
    plt.plot(subset['decomposition_level'], subset['modo_mixing'], 'o-', label=f'Noise={noise}')
plt.xlabel('Decomposition Level (K)')
plt.ylabel('Modo Mixing')
plt.title('VMD: Level Sensitivity')
plt.legend()

# VMD小波类型敏感性
plt.subplot(2, 3, 6)
vmd_sens_df.boxplot(column='modo_mixing', by='wavelet_type', ax=plt.gca())
plt.xlabel('Wavelet Type')
plt.ylabel('Modo Mixing')
plt.title('VMD: Wavelet Sensitivity')
plt.suptitle('')

plt.tight_layout()
plt.savefig('sensitivity_results/sensitivity_analysis.png')
plt.show()