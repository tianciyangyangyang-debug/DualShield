"""
简化版水印提取器 - 更快的训练速度

设计思路：
1. 简化的U-Net架构
2. 只使用MSE + SSIM损失（去掉感知损失）
3. 更少的参数，更快的训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """基础卷积块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)


class WatermarkExtractorSimple(nn.Module):
    """
    简化版水印提取器
    
    输入：带水印的生成图像 (B, 3, 512, 512)
    输出：提取的水印 (B, 3, 512, 512)
    """
    def __init__(self, base_channels=32):
        super().__init__()
        
        # 编码器
        self.enc1 = ConvBlock(3, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)
        
        # 瓶颈层
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)
        
        # 解码器
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, 2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        
        # 输出层
        self.output = nn.Sequential(
            nn.Conv2d(base_channels, 3, 1),
            nn.Sigmoid()  # 输出范围 [0, 1]
        )
        
    def forward(self, x):
        # 编码
        e1 = self.enc1(x)  # (B, 32, 512, 512)
        p1 = self.pool1(e1)  # (B, 32, 256, 256)
        
        e2 = self.enc2(p1)  # (B, 64, 256, 256)
        p2 = self.pool2(e2)  # (B, 64, 128, 128)
        
        e3 = self.enc3(p2)  # (B, 128, 128, 128)
        p3 = self.pool3(e3)  # (B, 128, 64, 64)
        
        # 瓶颈
        b = self.bottleneck(p3)  # (B, 256, 64, 64)
        
        # 解码（带跳跃连接）
        u3 = self.up3(b)  # (B, 128, 128, 128)
        u3 = torch.cat([u3, e3], dim=1)  # (B, 256, 128, 128)
        d3 = self.dec3(u3)  # (B, 128, 128, 128)
        
        u2 = self.up2(d3)  # (B, 64, 256, 256)
        u2 = torch.cat([u2, e2], dim=1)  # (B, 128, 256, 256)
        d2 = self.dec2(u2)  # (B, 64, 256, 256)
        
        u1 = self.up1(d2)  # (B, 32, 512, 512)
        u1 = torch.cat([u1, e1], dim=1)  # (B, 64, 512, 512)
        d1 = self.dec1(u1)  # (B, 32, 512, 512)
        
        # 输出
        out = self.output(d1)  # (B, 3, 512, 512)
        
        return out


def ssim_loss(x, y, window_size=11):
    """SSIM损失"""
    def gaussian_window(size, sigma=1.5):
        coords = torch.arange(size, dtype=torch.float32)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.outer(g)
    
    window = gaussian_window(window_size).to(x.device)
    window = window.expand(x.size(1), 1, window_size, window_size)
    
    mu1 = F.conv2d(x, window, padding=window_size//2, groups=x.size(1))
    mu2 = F.conv2d(y, window, padding=window_size//2, groups=y.size(1))
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(x * x, window, padding=window_size//2, groups=x.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(y * y, window, padding=window_size//2, groups=y.size(1)) - mu2_sq
    sigma12 = F.conv2d(x * y, window, padding=window_size//2, groups=x.size(1)) - mu1_mu2
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return 1 - ssim_map.mean()


class SimpleLoss(nn.Module):
    """简化的损失函数 - 只用MSE + SSIM"""
    def __init__(self, mse_weight=1.0, ssim_weight=1.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
    
    def forward(self, extracted_watermark, target_watermark):
        # MSE损失
        mse = F.mse_loss(extracted_watermark, target_watermark)
        
        # SSIM损失
        ssim = ssim_loss(extracted_watermark, target_watermark)
        
        # 总损失
        total = self.mse_weight * mse + self.ssim_weight * ssim
        
        return {
            'total': total,
            'mse': mse,
            'ssim': ssim
        }


def test_extractor():
    """测试提取器网络"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建网络
    extractor = WatermarkExtractorSimple().to(device)
    
    # 测试输入
    batch_size = 2
    generated_image = torch.randn(batch_size, 3, 512, 512).to(device)
    
    # 前向传播
    extracted_watermark = extractor(generated_image)
    
    print(f"Input shape: {generated_image.shape}")
    print(f"Output shape: {extracted_watermark.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in extractor.parameters())
    trainable_params = sum(p.numel() for p in extractor.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 测试损失函数
    target_watermark = torch.randn_like(extracted_watermark)
    loss_fn = SimpleLoss()
    losses = loss_fn(extracted_watermark, target_watermark)
    
    print(f"Loss breakdown: {losses}")


if __name__ == '__main__':
    test_extractor()
