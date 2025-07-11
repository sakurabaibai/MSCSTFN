import pywt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, hilbert
import matplotlib.pyplot as plt
import os
import time


# 自适应窗口大小函数
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


# 小波融合函数
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


# 信噪比计算函数
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


# 冲击强度计算函数
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
    end_idx = 0 + 1024 * 201
    X = X[start_idx:end_idx]
    Y = Y[start_idx:end_idx]
    Z = Z[start_idx:end_idx]

    # 定义噪声区域
    noise_region = slice(0, 100)  # 假设前100个点为噪声地板

    # 1. 使用默认小波基(db4)检测故障位置
    print("\n使用默认小波基(db4)检测故障位置...")
    _, win_sizes = rvwfm_fusion([X, Y, Z], wavelet='db4')

    # 对融合信号进行包络分析
    fused_default, _ = rvwfm_fusion([X, Y, Z], wavelet='db4')
    envelope = np.abs(hilbert(fused_default))
    peaks, _ = find_peaks(envelope, height=np.mean(envelope) + 3 * np.std(envelope), distance=50)
    print(f"检测到 {len(peaks)} 个潜在故障冲击点")

    # 2. 测试不同小波基的性能
    wavelets = ['db4', 'db6', 'db8', 'sym5', 'sym7', 'coif3', 'coif5', 'haar']
    results = []

    print("\n开始测试不同小波基性能...")
    for wavelet in wavelets:
        print(f"\n测试小波基: {wavelet}")

        # 记录开始时间
        start_time = time.time()

        # 执行融合
        fused_signal, _ = rvwfm_fusion([X, Y, Z], wavelet=wavelet)

        # 计算处理时间
        proc_time = time.time() - start_time

        # 计算信噪比
        snr = calculate_snr(fused_signal, noise_region)

        # 计算冲击强度
        impulse = calculate_impulse_strength(fused_signal, peaks)

        # 计算综合评分 (SNR占60%, 冲击强度占40%)
        # 归一化处理：假设SNR范围0-40dB，冲击强度0-1
        norm_snr = min(40, max(0, snr)) / 40
        norm_impulse = min(1, max(0, impulse))
        score = 0.6 * norm_snr + 0.4 * norm_impulse

        # 保存结果
        results.append({
            'wavelet': wavelet,
            'snr': snr,
            'impulse': impulse,
            'time': proc_time,
            'score': score
        })

        print(f"  - SNR: {snr:.2f} dB, 冲击强度: {impulse:.4f}, 耗时: {proc_time:.2f}秒")

    # 3. 选择最佳小波基
    best_wavelet = max(results, key=lambda x: x['score'])
    print(f"\n最佳小波基: {best_wavelet['wavelet']}")
    print(f"  SNR: {best_wavelet['snr']:.2f} dB")
    print(f"  冲击强度: {best_wavelet['impulse']:.4f}")
    print(f"  综合评分: {best_wavelet['score']:.3f}")
    print(f"  处理时间: {best_wavelet['time']:.2f}秒")

    # 4. 固定窗口融合（使用最佳小波基）
    fixed_start = time.time()
    fixed_fused, _ = rvwfm_fusion([X, Y, Z], wavelet=best_wavelet['wavelet'], min_window=4, max_window=4)
    fixed_end = time.time()
    fixed_time = fixed_end - fixed_start
    print(f"\n固定窗口融合时间: {fixed_time:.2f}秒")

    # 5. 自适应窗口融合（使用最佳小波基）
    adaptive_start = time.time()
    adaptive_fused, win_sizes = rvwfm_fusion([X, Y, Z], wavelet=best_wavelet['wavelet'], min_window=4, max_window=32)
    adaptive_end = time.time()
    adaptive_time = adaptive_end - adaptive_start
    print(f"自适应窗口融合时间: {adaptive_time:.2f}秒")

    # 计算信噪比和冲击强度
    snr_fixed = calculate_snr(fixed_fused, noise_region)
    snr_adaptive = calculate_snr(adaptive_fused, noise_region)
    impulse_fixed = calculate_impulse_strength(fixed_fused, peaks)
    impulse_adaptive = calculate_impulse_strength(adaptive_fused, peaks)

    print(f"\n固定窗口融合结果 - SNR: {snr_fixed:.2f} dB, 冲击强度: {impulse_fixed:.4f}")
    print(f"自适应窗口融合结果 - SNR: {snr_adaptive:.2f} dB, 冲击强度: {impulse_adaptive:.4f}")

    # 6. 可视化不同小波基性能比较
    wavelets = [r['wavelet'] for r in results]
    snrs = [r['snr'] for r in results]
    impulses = [r['impulse'] for r in results]
    scores = [r['score'] for r in results]
    times = [r['time'] for r in results]

    plt.figure(figsize=(15, 10))

    # SNR比较
    plt.subplot(2, 2, 1)
    plt.bar(wavelets, snrs, color='skyblue')
    plt.title('不同小波基的信噪比(SNR)比较')
    plt.ylabel('SNR (dB)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 冲击强度比较
    plt.subplot(2, 2, 2)
    plt.bar(wavelets, impulses, color='lightgreen')
    plt.title('不同小波基的冲击强度比较')
    plt.ylabel('冲击强度')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 综合评分比较
    plt.subplot(2, 2, 3)
    plt.bar(wavelets, scores, color='salmon')
    plt.title('不同小波基的综合评分比较')
    plt.ylabel('综合评分 (SNR60% + 冲击40%)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 处理时间比较
    plt.subplot(2, 2, 4)
    plt.bar(wavelets, times, color='gold')
    plt.title('不同小波基的处理时间比较')
    plt.ylabel('处理时间 (秒)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('wavelet_performance_comparison.png', dpi=300)
    plt.show()

    # 7. 固定窗口与自适应窗口融合结果比较
    plt.figure(figsize=(15, 8))

    # 原始信号
    plt.subplot(3, 1, 1)
    plt.plot(X, label='X-axis', alpha=0.7)
    plt.plot(Y, label='Y-axis', alpha=0.7)
    plt.plot(Z, label='Z-axis', alpha=0.7)
    plt.title(f'原始三轴振动信号 (使用最佳小波基: {best_wavelet["wavelet"]})')
    plt.legend()
    plt.grid(True)

    # 融合信号比较
    plt.subplot(3, 1, 2)
    plt.plot(fixed_fused, 'b-', label=f'固定窗口融合 (SNR: {snr_fixed:.2f} dB)')
    plt.plot(adaptive_fused, 'r-', label=f'自适应窗口融合 (SNR: {snr_adaptive:.2f} dB)')
    plt.title('固定窗口与自适应窗口融合结果比较')
    plt.legend()
    plt.grid(True)

    # 包络分析比较
    plt.subplot(3, 1, 3)
    plt.plot(np.abs(hilbert(fixed_fused)), 'b-', label='固定窗口包络')
    plt.plot(np.abs(hilbert(adaptive_fused)), 'r-', label='自适应窗口包络')
    plt.scatter(peaks, np.abs(hilbert(adaptive_fused))[peaks], c='g', s=50, zorder=5, label='故障冲击点')
    plt.title('包络分析比较')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('best_wavelet_comparison.png', dpi=300)
    plt.show()

    print("\n分析完成! 结果已保存为图像文件")