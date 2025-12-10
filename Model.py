import torch
import torch.nn as nn
import torch.nn.functional as F


class SAEA(nn.Module):
    """通道注意力机制 (SAEA) - 修复版本"""

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 确保中间层至少有1个神经元
        mid_channels = max(1, in_channels // reduction)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        # 平均池化路径
        avg_out = self.avg_pool(x).view(b, c)  # [B, C]
        avg_out = self.fc(avg_out).view(b, c, 1, 1)  # [B, C, 1, 1]

        # 最大池化路径
        max_out = self.max_pool(x).view(b, c)  # [B, C]
        max_out = self.fc(max_out).view(b, c, 1, 1)  # [B, C, 1, 1]

        # 结合两种池化结果
        return x * (avg_out + max_out)


class SpatialA(nn.Module):
    """空间注意力机制 (SpatialA)"""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通道维度上的平均和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # 拼接并卷积
        concat = torch.cat([avg_out, max_out], dim=1)
        att = self.sigmoid(self.conv(concat))
        return x * att


class SDPA(nn.Module):
    """缩放点积注意力 (Scaled Dot-Product Attention)"""

    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels, 1)
        self.key = nn.Conv2d(in_channels, in_channels, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.scale = in_channels ** -0.5

    def forward(self, x):
        B, C, H, W = x.size()

        # 生成Q, K, V
        Q = self.query(x).view(B, C, -1).permute(0, 2, 1)  # [B, N, C]
        K = self.key(x).view(B, C, -1)  # [B, C, N]
        V = self.value(x).view(B, C, -1).permute(0, 2, 1)  # [B, N, C]

        # 计算注意力分数
        attn_scores = torch.bmm(Q, K) * self.scale  # [B, N, N]
        attn_weights = self.softmax(attn_scores)

        # 应用注意力权重
        output = torch.bmm(attn_weights, V)  # [B, N, C]
        output = output.permute(0, 2, 1).view(B, C, H, W)
        return output


class CSFEM(nn.Module):
    """跨尺度特征提取模块"""

    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()

        # 分支1: 两个3x3卷积
        self.branch1_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.branch1_conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        # 分支2: 两个5x5卷积
        self.branch2_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 5, stride, padding=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.branch2_conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 5, padding=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        # 分支3: 两个7x7卷积
        self.branch3_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 7, stride, padding=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.branch3_conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 7, padding=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        # 注意力机制（每个分支独立）
        self.ca1 = SAEA(out_channels)
        self.sa1 = SpatialA()
        self.ca2 = SAEA(out_channels)
        self.sa2 = SpatialA()
        self.ca3 = SAEA(out_channels)
        self.sa3 = SpatialA()

        # 缩放点积注意力
        self.sdpa = SDPA(out_channels)

        # 分支融合（1x1卷积）
        self.conv_fuse = nn.Conv2d(out_channels * 3, out_channels, 1)

    def forward(self, x):
        # 分支1处理
        x1_conv1 = self.branch1_conv1(x)  # 第一个3x3卷积
        x1 = self.branch1_conv2(x1_conv1)  # 第二个3x3卷积
        x1 = self.ca1(x1)  # 通道注意力
        x1 = self.sa1(x1)  # 空间注意力

        # 分支2处理（跨尺度连接）
        x2_conv1 = self.branch2_conv1(x)  # 第一个5x5卷积
        x2_conv1 = x2_conv1 + x1_conv1  # 跨尺度连接：添加分支1的第一个卷积输出
        x2 = self.branch2_conv2(x2_conv1)  # 第二个5x5卷积
        x2 = self.ca2(x2)  # 通道注意力
        x2 = self.sa2(x2)  # 空间注意力

        # 分支3处理（跨尺度连接）
        x3_conv1 = self.branch3_conv1(x)  # 第一个7x7卷积
        x3_conv1 = x3_conv1 + x2_conv1  # 跨尺度连接：添加分支2的第一个卷积输出
        x3 = self.branch3_conv2(x3_conv1)  # 第二个7x7卷积
        x3 = self.ca3(x3)  # 通道注意力
        x3 = self.sa3(x3)  # 空间注意力

        # 多分支融合
        fused = torch.cat([x1, x2, x3], dim=1)  # 通道维度拼接
        fused = self.conv_fuse(fused)  # 1x1卷积融合

        # 缩放点积注意力
        output = self.sdpa(fused)
        return output


class DSDRM_Block(nn.Module):
    """深度可分离残差基础块"""

    def __init__(self, in_channels, out_channels, stride, expansion=6):
        super().__init__()
        self.expanded_channels = in_channels * expansion

        # 扩展层 (1x1卷积)
        self.expand_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.expanded_channels, 1, bias=False),
            nn.BatchNorm2d(self.expanded_channels),
            nn.ReLU6(inplace=True)
        )

        # 深度卷积层 (3x3卷积)
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(self.expanded_channels, self.expanded_channels, 3,
                      stride, padding=1, groups=self.expanded_channels, bias=False),
            nn.BatchNorm2d(self.expanded_channels),
            nn.ReLU6(inplace=True)
        )

        # 压缩层 (1x1卷积)
        self.project_conv = nn.Sequential(
            nn.Conv2d(self.expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # 残差连接条件
        self.use_residual = stride == 1 and in_channels == out_channels
        if not self.use_residual:
            # 如果不满足残差条件，添加一个1x1卷积调整通道和尺寸
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        # 扩展通道
        x = self.expand_conv(x)
        # 深度卷积
        x = self.depthwise_conv(x)
        # 压缩通道
        x = self.project_conv(x)

        # 残差连接
        if self.use_residual:
            return x + identity
        else:
            return x + self.shortcut(identity)


class DSDRM(nn.Module):
    """深度可分离双残差模块"""

    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        # 第一个块 (步长为2，用于下采样)
        self.block1 = DSDRM_Block(in_channels, out_channels, stride)

        # 第二个块 (步长为1)
        self.block2 = DSDRM_Block(out_channels, out_channels, 1)

        # 双残差路径
        # 第一个残差路径（用于第一个块）
        self.res_path1 = nn.Sequential(
            nn.MaxPool2d(3, stride, padding=1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # 第二个残差路径（用于第二个块）
        self.res_path2 = nn.Sequential(
            nn.MaxPool2d(3, 1, padding=1),  # 步长改为1，保持尺寸不变
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        # 第一个残差路径
        res1 = self.res_path1(x)

        # 第一个块处理
        x1 = self.block1(x)

        # 添加第一个残差连接
        x1 = x1 + res1

        # 第二个残差路径（从x1开始）
        res2 = self.res_path2(x1)

        # 第二个块处理
        x2 = self.block2(x1)

        # 添加第二个残差连接
        output = x2 + res2

        return output


# ================== 主网络架构 ==================
class MSCSTFN(nn.Module):
    """多传感器信息融合跨尺度时频网络"""

    def __init__(self, num_classes=10):
        super().__init__()
        # 模块配置
        self.csfem1 = CSFEM(1, 12, stride=2)  # 输出: 112x112x12
        self.dsdrm1 = DSDRM(12, 24, stride=2)  # 输出: 56x56x24
        self.csfem2 = CSFEM(24, 96, stride=2)  # 输出: 28x28x96
        self.dsdrm2 = DSDRM(96, 192, stride=2)  # 输出: 14x14x192
        self.dsdrm3 = DSDRM(192, 384, stride=2)  # 输出: 7x7x384

        # 连接层 (1x1卷积)
        self.connect = nn.Conv2d(384, 762, 1)  # 输出: 7x7x762

        # 分类层
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 输出: 1x1x762
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(762, num_classes)

    def forward(self, x):
        # 输入x: [B, 1, 224, 224] (时频图像)
        x = self.csfem1(x)  # [B, 12, 112, 112]
        x = self.dsdrm1(x)  # [B, 24, 56, 56]
        x = self.csfem2(x)  # [B, 96, 28, 28]
        x = self.dsdrm2(x)  # [B, 192, 14, 14]
        x = self.dsdrm3(x)  # [B, 384, 7, 7]
        x = self.connect(x)  # [B, 762, 7, 7]
        x = self.avgpool(x)  # [B, 762, 1, 1]
        x = torch.flatten(x, 1)  # [B, 762]
        x = self.dropout(x)
        x = self.classifier(x)  # [B, num_classes]
        return x


# ================== 使用示例 ==================
if __name__ == "__main__":
    # 模型初始化
    model = MSCSTFN(num_classes=10)

    # 模拟输入 (时频图像)
    input_tensor = torch.randn(4, 1, 224, 224)  # [batch, channel, height, width]

    # 前向传播
    output = model(input_tensor)

    print(f"Output shape: {output.shape}")  # 预期: [4, 10]
