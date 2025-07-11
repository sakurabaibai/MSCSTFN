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

    # 重构融合信号
    fused_coeffs = [fused_low] + fused_high_list
    fused_signal = pywt.waverec(fused_coeffs, wavelet)

    return fused_signal, window_sizes


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
    end_idx = 0 + 1025 * 201
    X = X[start_idx:end_idx]
    Y = Y[start_idx:end_idx]
    Z = Z[start_idx:end_idx]

    # 自适应窗口融合
    adaptive_start = time.time()
    adaptive_fused, win_sizes = rvwfm_fusion([X, Y, Z], min_window=4, max_window=32)
    adaptive_end = time.time()
    adaptive_time = adaptive_end - adaptive_start
    print(f"自适应融合完成! 耗时: {adaptive_time:.2f}秒")

    # 创建保存目录
    output_dir = "fusion_results"
    os.makedirs(output_dir, exist_ok=True)

    # 保存融合信号
    output_path = os.path.join(output_dir, "fused_signal.csv")
    pd.DataFrame(adaptive_fused).to_csv(output_path, index=False, header=False)
    print(f"融合信号已保存至: {output_path}")

