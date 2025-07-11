import pywt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, hilbert
import matplotlib.pyplot as plt
import os
import time


def adaptive_window_size(coeff_segment, min_size=4, max_size=32):
    """基于局部瞬态冲击特征的自适应窗口算法"""
    if len(coeff_segment) < min_size:
        return min_size

    # 检测显著峰值（瞬态冲击特征）
    abs_coeff = np.abs(coeff_segment)
    if np.all(abs_coeff == 0):
        return max_size

    # 动态阈值设置
    median_val = np.median(abs_coeff)
    std_val = np.std(abs_coeff)
    peak_threshold = median_val + 3 * std_val

    # 查找显著峰值
    peaks, _ = find_peaks(abs_coeff, height=peak_threshold, distance=5)

    # 计算冲击特征密度
    peak_density = len(peaks) / len(coeff_segment) if len(coeff_segment) > 0 else 0

    # 根据冲击密度确定窗口大小
    if peak_density > 0.15:  # 高冲击密度（故障区域）
        return min_size
    elif peak_density > 0.05:  # 中等冲击密度
        return min(12, max_size)
    else:  # 平稳区域
        return max_size


def rvwfm_fusion(signals, wavelet='db4', level=6, min_window=4, max_window=32):
    """改进的自适应窗口RVWFM融合"""
    # 小波分解
    coeffs_list = [pywt.wavedec(s, wavelet, level=level) for s in signals]

    # 低频系数融合
    fused_low = np.mean([c[0] for c in coeffs_list], axis=0)

    # 高频系数融合（自适应窗口）
    fused_high_list = []
    window_sizes = []  # 记录窗口大小变化

    for i in range(1, level + 1):
        layer_coeffs = [c[i] for c in coeffs_list]
        fused_high = np.zeros_like(layer_coeffs[0])
        level_window_sizes = []

        for pos in range(len(fused_high)):
            # 分析局部区域（当前点±50点）
            start = max(0, pos - 50)
            end = min(len(fused_high), pos + 50)
            segment = np.concatenate([lc[start:end] for lc in layer_coeffs])

            # 计算自适应窗口大小
            win_size = adaptive_window_size(segment, min_window, max_window)
            level_window_sizes.append(win_size)

            # 应用窗口
            win_start = max(0, pos - win_size // 2)
            win_end = min(len(fused_high), pos + win_size // 2 + 1)

            # 选择最大方差信号
            variances = []
            for lc in layer_coeffs:
                window = lc[win_start:win_end]
                variances.append(np.var(window))

            fused_high[pos] = layer_coeffs[np.argmax(variances)][pos]

        fused_high_list.append(fused_high)
        window_sizes.append(level_window_sizes)

    # 信号重构
    fused_coeffs = [fused_low] + fused_high_list
    reconstructed = pywt.waverec(fused_coeffs, wavelet)
    # 确保信号长度与原始信号一致
    return reconstructed[:len(signals[0])], window_sizes


def calculate_snr(signal, noise_floor_region):
    """计算信号的信噪比(SNR)"""
    # 提取噪声地板区域
    noise = signal[noise_floor_region]

    # 计算噪声功率
    noise_power = np.mean(noise ** 2)

    # 计算整体信号功率
    signal_power = np.mean(signal ** 2)

    # 计算信噪比(dB)
    snr = 10 * np.log10((signal_power - noise_power) / noise_power)
    return snr


def calculate_impulse_strength(signal, fault_positions, window=20):
    """计算故障冲击强度"""
    strengths = []
    for pos in fault_positions:
        if pos < len(signal) - window:
            segment = signal[pos - window // 2:pos + window // 2]
            envelope = np.abs(hilbert(segment))
            strengths.append(np.max(envelope))
    return np.mean(strengths) if strengths else 0


# 主程序
if __name__ == "__main__":
    # 加载实际齿轮故障数据
    file_path = 'gearset/Chipped_20_0.csv'

    if not os.path.exists(file_path):
        print(f"错误: 文件 '{file_path}' 不存在!")
        exit()

    try:
        data = pd.read_csv(file_path, sep='\t', header=None)
    except Exception as e:
        print(f"读取文件错误: {e}")
        exit()

    # 提取三轴振动信号
    try:
        # 假设数据格式为三列振动信号
        X = data.iloc[:, 0].astype(float).values
        Y = data.iloc[:, 1].astype(float).values
        Z = data.iloc[:, 2].astype(float).values
    except Exception as e:
        print(f"数据提取错误: {e}")
        print("尝试备用提取方法...")
        try:
            # 尝试正则表达式提取
            Xx = data[0].str.extract(r'\t(.*?)\t(.*?)\t(.*?)\t', expand=False)
            X = Xx.iloc[:, 0].astype(float).values
            Y = Xx.iloc[:, 1].astype(float).values
            Z = Xx.iloc[:, 2].astype(float).values
        except:
            print("无法提取三轴信号，请检查数据格式")
            exit()

    print(f"成功加载数据: X={len(X)}, Y={len(Y)}, Z={len(Z)} 个采样点")

    # 截取部分信号进行分析（避免过长处理时间）
    start_idx = 0
    end_idx = 0+1024*201
    X = X[start_idx:end_idx]
    Y = Y[start_idx:end_idx]
    Z = Z[start_idx:end_idx]

    # 固定窗口融合
    fixed_start = time.time()
    fixed_fused, _ = rvwfm_fusion([X, Y, Z], min_window=4, max_window=4)
    fixed_end = time.time()
    fixed_time = fixed_end - fixed_start
    print(f"固定窗口融合时间: {fixed_time}秒")


    # 自适应窗口融合
    adaptive_start = time.time()
    adaptive_fused, win_sizes = rvwfm_fusion([X, Y, Z], min_window=4, max_window=32)
    adaptive_end = time.time()
    adaptive_time = adaptive_end - adaptive_start
    print(f"自适应窗口融合时间: {adaptive_time}秒")
    # 计算信噪比和冲击强度
    noise_region = slice(0, 100)  # 假设前100个点为噪声地板
    snr_fixed = calculate_snr(fixed_fused, noise_region)
    snr_adaptive = calculate_snr(adaptive_fused, noise_region)

    # 通过包络分析检测故障位置
    envelope = np.abs(hilbert(adaptive_fused))
    peaks, _ = find_peaks(envelope, height=np.mean(envelope) + 3 * np.std(envelope), distance=50)

    impulse_fixed = calculate_impulse_strength(fixed_fused, peaks)
    impulse_adaptive = calculate_impulse_strength(adaptive_fused, peaks)

    # 创建结果目录
    os.makedirs('result', exist_ok=True)

    # 可视化结果
    plt.figure(figsize=(15, 15))

    # 原始信号
    plt.subplot(5, 1, 1)
    plt.plot(X, label='X-axis', alpha=0.7)
    plt.plot(Y, label='Y-axis', alpha=0.7)
    plt.plot(Z, label='Z-axis', alpha=0.7)
    plt.title(f'Actual Gear Vibration Signals (Chipped Tooth) - {len(X)} samples')
    plt.legend()
    plt.grid(True)

    # 固定窗口融合
    plt.subplot(5, 1, 2)
    plt.plot(fixed_fused, 'b-')
    plt.title(f'Fused Signal (Fixed Window=4) - SNR: {snr_fixed:.2f} dB, Impulse: {impulse_fixed:.4f}')
    plt.grid(True)

    # 自适应窗口融合
    plt.subplot(5, 1, 3)
    plt.plot(adaptive_fused, 'r-')
    plt.title(f'Fused Signal (Adaptive Window) - SNR: {snr_adaptive:.2f} dB, Impulse: {impulse_adaptive:.4f}')
    plt.grid(True)

    # 差异信号
    plt.subplot(5, 1, 4)
    diff = adaptive_fused - fixed_fused
    plt.plot(diff, 'g-')
    plt.title('Difference: Adaptive - Fixed Window')
    plt.grid(True)

    # 窗口大小变化（以第3层小波系数为例）
    plt.subplot(5, 1, 5)
    plt.plot(win_sizes[2], 'c-')
    plt.title('Adaptive Window Size Variation (Level 3 Coefficients)')
    plt.xlabel('Coefficient Position')
    plt.ylabel('Window Size')
    plt.ylim(0, 35)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('results/actual_gear_fusion_comparison.png', dpi=300)

    # 故障点区域放大分析
    if len(peaks) > 0:
        # 选择最强的三个冲击点
        strongest_peaks = sorted(peaks, key=lambda i: envelope[i], reverse=True)[:3]

        for i, peak in enumerate(strongest_peaks):
            plt.figure(figsize=(12, 8))
            fault_region = slice(max(0, peak - 100), min(len(adaptive_fused), peak + 100))

            plt.subplot(3, 1, 1)
            plt.plot(X[fault_region], 'b-', label='X-axis')
            plt.plot(Y[fault_region], 'g-', label='Y-axis')
            plt.plot(Z[fault_region], 'r-', label='Z-axis')
            plt.axvline(x=peak - fault_region.start, color='k', linestyle='--', alpha=0.5)
            plt.title(f'Original Signals at Fault Region {i + 1} (Position: {peak})')
            plt.legend()
            plt.grid(True)

            plt.subplot(3, 1, 2)
            plt.plot(fixed_fused[fault_region], 'b-', label='Fixed Window')
            plt.plot(adaptive_fused[fault_region], 'r-', label='Adaptive Window')
            plt.axvline(x=peak - fault_region.start, color='k', linestyle='--', alpha=0.5)
            plt.title('Fused Signals Comparison')
            plt.legend()
            plt.grid(True)

            plt.subplot(3, 1, 3)
            plt.plot(win_sizes[2][fault_region], 'm-')
            plt.axvline(x=peak - fault_region.start, color='k', linestyle='--', alpha=0.5)
            plt.title('Adaptive Window Size at Fault Region')
            plt.xlabel('Coefficient Position')
            plt.ylabel('Window Size')
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(f'results/fault_region_{i + 1}_comparison.png', dpi=300)

    # 包络分析对比
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(np.abs(hilbert(fixed_fused)), 'b-', label='Fixed Window')
    plt.plot(np.abs(hilbert(adaptive_fused)), 'r-', label='Adaptive Window')
    plt.scatter(peaks, envelope[peaks], c='g', s=50, zorder=5, label='Fault Peaks')
    plt.title('Envelope Analysis of Fused Signals')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(np.abs(hilbert(adaptive_fused)) - np.abs(hilbert(fixed_fused)), 'm-')
    plt.title('Envelope Difference: Adaptive - Fixed Window')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('results/envelope_comparison.png', dpi=300)

    print("分析完成! 结果已保存到results目录")
    plt.show()