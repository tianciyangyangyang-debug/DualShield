"""
水印提取器网络 - 从DreamBooth生成的图像中提取水印

设计思路：
1. 输入：512x512 RGB图像（DreamBooth生成，带水印特征）
2. 输出：512x512 RGB水印图像
3. 架构：U-Net编码器-解码器，专注于提取水印特征
4. 损失：提取水印与原始水印的相似度（MSE + SSIM + 感知损失）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19


class SimpleGate(nn.Module):
    """SimpleGate激活函数 - 来自NAFNet"""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class LayerNorm2d(nn.Module):
    """2D LayerNorm"""
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ExtractorBlock(nn.Module):
    """提取器基础块"""
    def __init__(self, channels, dw_expand=2, ffn_expand=2):
        super().__init__()
        
        # Depthwise Separable Conv
        dw_channels = channels * dw_expand
        self.conv1 = nn.Conv2d(channels, dw_channels, 1)
        self.conv2 = nn.Conv2d(dw_channels, dw_channels, 3, 1, 1, groups=dw_channels)
        # SimpleGate会将通道数减半，所以conv3的输入是dw_channels//2
        self.conv3 = nn.Conv2d(dw_channels // 2, channels, 1)
        
        # SimpleGate
        self.sg = SimpleGate()
        
        # LayerNorm
        self.norm1 = LayerNorm2d(channels)
        self.norm2 = LayerNorm2d(channels)
        
        # FFN
        ffn_channels = channels * ffn_expand
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, ffn_channels * 2, 1),
            SimpleGate(),
            nn.Conv2d(ffn_channels, channels, 1)
        )
        
        # Dropout
        self.dropout1 = nn.Dropout2d(0.1)
        self.dropout2 = nn.Dropout2d(0.1)
        
    def forward(self, x):
        # 第一个分支：Depthwise Separable Conv
        inp = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)  # SimpleGate会将通道数减半
        x = self.conv3(x)  # 所以conv3的输入通道应该是dw_channels//2
        x = self.dropout1(x)
        x = x + inp
        
        # 第二个分支：FFN
        inp = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout2(x)
        x = x + inp
        
        return x


class WatermarkExtractor(nn.Module):
    """
    水印提取器网络
    
    架构：U-Net编码器-解码器
    - 编码器：逐步下采样，提取多尺度特征
    - 解码器：逐步上采样，重建水印
    - 跳跃连接：保留细节信息
    """
    
    def __init__(self, in_channels=3, out_channels=3, base_channels=64):
        super().__init__()
        
        # 输入投影
        self.input_proj = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        
        # 编码器 (下采样)
        self.enc1 = nn.Sequential(
            *[ExtractorBlock(base_channels) for _ in range(2)]
        )
        self.down1 = nn.Conv2d(base_channels, base_channels*2, 3, 2, 1)
        
        self.enc2 = nn.Sequential(
            *[ExtractorBlock(base_channels*2) for _ in range(2)]
        )
        self.down2 = nn.Conv2d(base_channels*2, base_channels*4, 3, 2, 1)
        
        self.enc3 = nn.Sequential(
            *[ExtractorBlock(base_channels*4) for _ in range(2)]
        )
        self.down3 = nn.Conv2d(base_channels*4, base_channels*8, 3, 2, 1)
        
        # 瓶颈层
        self.bottleneck = nn.Sequential(
            *[ExtractorBlock(base_channels*8) for _ in range(4)]
        )
        
        # 解码器 (上采样)
        self.up3 = nn.ConvTranspose2d(base_channels*8, base_channels*4, 3, 2, 1, 1)
        self.dec3 = nn.Sequential(
            nn.Conv2d(base_channels*8, base_channels*4, 1),  # 融合跳跃连接
            *[ExtractorBlock(base_channels*4) for _ in range(2)]
        )
        
        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 3, 2, 1, 1)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*2, 1),  # 融合跳跃连接
            *[ExtractorBlock(base_channels*2) for _ in range(2)]
        )
        
        self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, 3, 2, 1, 1)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels, 1),  # 融合跳跃连接
            *[ExtractorBlock(base_channels) for _ in range(2)]
        )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Conv2d(base_channels, base_channels//2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels//2, out_channels, 3, 1, 1),
            nn.Sigmoid()  # 输出范围[0,1]
        )
        
    def forward(self, x):
        # 输入投影
        x = self.input_proj(x)
        
        # 编码器
        enc1 = self.enc1(x)
        x = self.down1(enc1)
        
        enc2 = self.enc2(x)
        x = self.down2(enc2)
        
        enc3 = self.enc3(x)
        x = self.down3(enc3)
        
        # 瓶颈层
        x = self.bottleneck(x)
        
        # 解码器
        x = self.up3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)
        
        x = self.up2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        
        x = self.up1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)
        
        # 输出投影
        watermark = self.output_proj(x)
        
        return watermark


class PerceptualLoss(nn.Module):
    """感知损失 - 使用VGG19特征"""
    def __init__(self, device='cuda'):
        super().__init__()
        vgg = vgg19(pretrained=True).features
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])   # relu1_2
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])  # relu2_2
        self.slice3 = nn.Sequential(*list(vgg.children())[9:18]) # relu3_4
        
        # 移动到指定设备
        self.slice1 = self.slice1.to(device)
        self.slice2 = self.slice2.to(device)
        self.slice3 = self.slice3.to(device)
        
        # 冻结VGG参数
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x, y):
        # 归一化到ImageNet范围
        x = (x - 0.5) * 2  # [0,1] -> [-1,1]
        y = (y - 0.5) * 2
        
        # 提取特征
        x1, y1 = self.slice1(x), self.slice1(y)
        x2, y2 = self.slice2(x1), self.slice2(y1)
        x3, y3 = self.slice3(x2), self.slice3(y2)
        
        # 计算损失
        loss = F.mse_loss(x1, y1) + F.mse_loss(x2, y2) + F.mse_loss(x3, y3)
        return loss


class ExtractorLoss(nn.Module):
    """提取器综合损失函数"""
    def __init__(self, mse_weight=1.0, ssim_weight=0.5, perceptual_weight=0.1, device='cuda'):
        super().__init__()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight
        self.perceptual_loss = PerceptualLoss(device=device)
        
    def ssim_loss(self, x, y, window_size=11):
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
    
    def forward(self, extracted_watermark, target_watermark):
        # MSE损失
        mse_loss = F.mse_loss(extracted_watermark, target_watermark)
        
        # SSIM损失
        ssim_loss = self.ssim_loss(extracted_watermark, target_watermark)
        
        # 感知损失
        perceptual_loss = self.perceptual_loss(extracted_watermark, target_watermark)
        
        # 总损失
        total_loss = (self.mse_weight * mse_loss + 
                     self.ssim_weight * ssim_loss + 
                     self.perceptual_weight * perceptual_loss)
        
        return {
            'total': total_loss,
            'mse': mse_loss,
            'ssim': ssim_loss,
            'perceptual': perceptual_loss
        }


def test_extractor():
    """测试提取器网络"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建网络
    extractor = WatermarkExtractor().to(device)
    
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
    loss_fn = ExtractorLoss()
    losses = loss_fn(extracted_watermark, target_watermark)
    
    print(f"Loss breakdown: {losses}")


if __name__ == '__main__':
    test_extractor()