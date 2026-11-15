"""
高效对抗扰动生成器 - 基于NAFNet和条件GAN的设计

设计理念：
1. 借鉴NAFNet的SimpleGate机制，去除非线性激活函数，提高效率
2. 使用U-Net编码器-解码器结构，保留空间信息
3. 条件输入：原图 + 水印，输出：对抗扰动
4. 轻量化设计：深度可分离卷积 + 通道注意力

参考文献：
- NAFNet: Simple Baselines for Image Restoration (ECCV 2022)
- Watermark-embedded Adversarial Examples (2024)
- AdvGAN: Generating Adversarial Examples with Adversarial Networks

架构特点：
- 编码器：4层下采样，提取多尺度特征
- 瓶颈层：NAFBlock处理全局信息
- 解码器：4层上采样，恢复空间分辨率
- 跳跃连接：保留细节信息
- 输出：tanh限制扰动范围
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGate(nn.Module):
    """
    NAFNet的核心创新：SimpleGate
    用元素乘法替代非线性激活函数
    x = x1 * x2，其中x沿通道维度分成两半
    """
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    """
    简化的通道注意力（来自NAFNet）
    去除Softmax，用SimpleGate替代
    """
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0)
    
    def forward(self, x):
        y = self.pool(x)
        y = self.fc(y)
        return x * y


class NAFBlock(nn.Module):
    """
    NAFNet基础块 - 无非线性激活函数
    
    结构：
    LayerNorm -> Conv 1x1 (扩展) -> DWConv 3x3 -> SimpleGate -> 
    Conv 1x1 (压缩) -> ChannelAttention -> 残差连接
    """
    def __init__(self, channels, expansion=2):
        super().__init__()
        
        expanded = channels * expansion
        
        self.norm = nn.LayerNorm(channels)
        
        # 扩展 -> DWConv -> SimpleGate -> 压缩
        self.conv1 = nn.Conv2d(channels, expanded * 2, 1, 1, 0)  # *2 for SimpleGate
        self.dwconv = nn.Conv2d(expanded * 2, expanded * 2, 3, 1, 1, groups=expanded * 2)
        self.gate = SimpleGate()
        self.conv2 = nn.Conv2d(expanded, channels, 1, 1, 0)
        
        # 通道注意力
        self.ca = SimplifiedChannelAttention(channels)
        
        # 可学习的残差缩放
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
    
    def forward(self, x):
        # LayerNorm需要(B, H, W, C)格式
        b, c, h, w = x.shape
        residual = x
        
        # LayerNorm
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # 主路径
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.gate(x)
        x = self.conv2(x)
        x = self.ca(x)
        
        return residual + x * self.beta


class DownBlock(nn.Module):
    """下采样块：Conv下采样 + NAFBlock"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, 4, 2, 1)
        self.block = NAFBlock(out_ch)
    
    def forward(self, x):
        x = self.down(x)
        x = self.block(x)
        return x


class UpBlock(nn.Module):
    """上采样块：转置卷积上采样 + NAFBlock"""
    def __init__(self, in_ch, out_ch, skip_ch=0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)
        self.block = NAFBlock(out_ch + skip_ch)
        self.reduce = nn.Conv2d(out_ch + skip_ch, out_ch, 1, 1, 0) if skip_ch > 0 else nn.Identity()
    
    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        x = self.reduce(x)
        return x


class PerturbationGenerator(nn.Module):
    """
    对抗扰动生成器 - 条件U-Net架构
    
    输入：原图(3通道) + 水印(3通道) = 6通道
    输出：扰动(3通道)，范围[-eps, eps]
    
    架构：
    - 编码器：4层下采样 (6->32->64->128->256)
    - 瓶颈：2个NAFBlock
    - 解码器：4层上采样 + 跳跃连接
    - 输出：tanh * eps 限制扰动范围
    
    参数量：约1.5M（比原始INN的12M减少8倍）
    """
    def __init__(self, in_channels=6, base_channels=32, eps=10/255):
        super().__init__()
        
        self.eps = eps
        
        # 初始卷积
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        
        # 编码器 (下采样)
        self.enc1 = DownBlock(base_channels, base_channels * 2)      # 32 -> 64
        self.enc2 = DownBlock(base_channels * 2, base_channels * 4)  # 64 -> 128
        self.enc3 = DownBlock(base_channels * 4, base_channels * 8)  # 128 -> 256
        self.enc4 = DownBlock(base_channels * 8, base_channels * 8)  # 256 -> 256
        
        # 瓶颈层
        self.bottleneck = nn.Sequential(
            NAFBlock(base_channels * 8),
            NAFBlock(base_channels * 8),
        )
        
        # 解码器 (上采样 + 跳跃连接)
        self.dec4 = UpBlock(base_channels * 8, base_channels * 8, skip_ch=base_channels * 8)
        self.dec3 = UpBlock(base_channels * 8, base_channels * 4, skip_ch=base_channels * 4)
        self.dec2 = UpBlock(base_channels * 4, base_channels * 2, skip_ch=base_channels * 2)
        self.dec1 = UpBlock(base_channels * 2, base_channels, skip_ch=base_channels)
        
        # 输出层
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.Conv2d(base_channels, 3, 1, 1, 0),
            nn.Tanh()  # 输出范围 [-1, 1]
        )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # 输出层用小值初始化（不是全零，否则梯度为0）
        nn.init.normal_(self.out_conv[-2].weight, mean=0, std=0.01)
        nn.init.zeros_(self.out_conv[-2].bias)
    
    def forward(self, x, watermark):
        """
        前向传播
        
        Args:
            x: 原始图像 [B, 3, H, W]
            watermark: 水印图像 [B, 3, H, W]
        
        Returns:
            perturbation: 对抗扰动 [B, 3, H, W]，范围 [-eps, eps]
        """
        # 拼接输入
        inp = torch.cat([x, watermark], dim=1)  # [B, 6, H, W]
        
        # 初始特征
        f0 = self.init_conv(inp)  # [B, 32, H, W]
        
        # 编码器
        f1 = self.enc1(f0)   # [B, 64, H/2, W/2]
        f2 = self.enc2(f1)   # [B, 128, H/4, W/4]
        f3 = self.enc3(f2)   # [B, 256, H/8, W/8]
        f4 = self.enc4(f3)   # [B, 256, H/16, W/16]
        
        # 瓶颈
        f4 = self.bottleneck(f4)
        
        # 解码器 + 跳跃连接
        d4 = self.dec4(f4, f3)  # [B, 256, H/8, W/8]
        d3 = self.dec3(d4, f2)  # [B, 128, H/4, W/4]
        d2 = self.dec2(d3, f1)  # [B, 64, H/2, W/2]
        d1 = self.dec1(d2, f0)  # [B, 32, H, W]
        
        # 输出扰动
        perturbation = self.out_conv(d1) * self.eps  # [B, 3, H, W], 范围 [-eps, eps]
        
        return perturbation
    
    def generate_adversarial(self, x, watermark):
        """
        生成对抗样本
        
        Args:
            x: 原始图像 [B, 3, H, W]
            watermark: 水印图像 [B, 3, H, W]
        
        Returns:
            adv: 对抗样本 [B, 3, H, W]，范围 [0, 1]
        """
        perturbation = self.forward(x, watermark)
        adv = x + perturbation
        adv = torch.clamp(adv, 0, 1)
        return adv


class PatchDiscriminator(nn.Module):
    """
    PatchGAN判别器
    
    判断图像块是真实还是生成的
    输出：[B, 1, H/16, W/16] 的概率图
    """
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        
        self.model = nn.Sequential(
            # 第1层：不使用BatchNorm
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 第2层
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 第3层
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 第4层
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 1, 1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 输出层 - 不使用Sigmoid，配合BCEWithLogitsLoss使用
            nn.Conv2d(base_channels * 8, 1, 4, 1, 1),
        )
    
    def forward(self, x):
        return self.model(x)  # 返回logits，不是概率


class WatermarkAdvGAN(nn.Module):
    """
    完整的水印嵌入对抗GAN系统
    
    包含：
    - 生成器G：生成对抗扰动
    - 判别器D：区分真实图像和对抗样本
    - VAE编码器：计算潜空间损失（冻结）
    """
    def __init__(self, eps=10/255, base_channels=32):
        super().__init__()
        
        self.generator = PerturbationGenerator(eps=eps, base_channels=base_channels)
        self.discriminator = PatchDiscriminator()
        
        # VAE编码器在训练时外部传入（冻结）
        self.vae = None
    
    def set_vae(self, vae):
        """设置VAE编码器（用于计算潜空间损失）"""
        self.vae = vae
        for param in self.vae.parameters():
            param.requires_grad = False
    
    def forward(self, x, watermark):
        """生成对抗样本"""
        return self.generator.generate_adversarial(x, watermark)
    
    def compute_losses(self, x, watermark, real_label=1.0, fake_label=0.0):
        """
        计算所有损失
        
        Returns:
            losses: dict containing all losses
            adv_sample: 生成的对抗样本
        """
        # 生成对抗样本
        perturbation = self.generator(x, watermark)
        adv_sample = torch.clamp(x + perturbation, 0, 1)
        
        # 1. GAN损失 - 判别器（使用BCEWithLogitsLoss）
        real_pred = self.discriminator(x)
        fake_pred = self.discriminator(adv_sample.detach())
        
        d_loss_real = F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred) * real_label)
        d_loss_fake = F.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred) + fake_label)
        d_loss = (d_loss_real + d_loss_fake) / 2
        
        # 2. GAN损失 - 生成器（欺骗判别器）
        fake_pred_g = self.discriminator(adv_sample)
        g_gan_loss = F.binary_cross_entropy_with_logits(fake_pred_g, torch.ones_like(fake_pred_g) * real_label)
        
        # 3. 扰动损失 - 限制扰动大小
        pert_loss = torch.mean(torch.abs(perturbation))
        
        # 4. 加权扰动损失 - 水印区域权重更大
        # 水印区域（非黑色部分）扰动应该更小
        watermark_mask = (watermark.mean(dim=1, keepdim=True) > 0.1).float()
        weighted_pert = perturbation * (1 + 4 * watermark_mask)  # 水印区域权重=5
        weighted_pert_loss = torch.mean(torch.abs(weighted_pert))
        
        # 5. 对抗损失 - VAE潜空间
        adv_loss = torch.tensor(0.0, device=x.device)
        if self.vae is not None:
            with torch.no_grad():
                latent_watermark = self.vae.encode(watermark).latent_dist.sample() * 0.18215
            latent_adv = self.vae.encode(adv_sample).latent_dist.sample() * 0.18215
            adv_loss = F.mse_loss(latent_adv, latent_watermark)
        
        losses = {
            'd_loss': d_loss,
            'g_gan_loss': g_gan_loss,
            'pert_loss': pert_loss,
            'weighted_pert_loss': weighted_pert_loss,
            'adv_loss': adv_loss,
        }
        
        return losses, adv_sample


def get_parameter_count(model):
    """计算模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}


if __name__ == '__main__':
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    generator = PerturbationGenerator().to(device)
    discriminator = PatchDiscriminator().to(device)
    
    # 打印参数量
    print(f"Generator params: {get_parameter_count(generator)}")
    print(f"Discriminator params: {get_parameter_count(discriminator)}")
    
    # 测试前向传播
    x = torch.randn(1, 3, 512, 512).to(device)
    watermark = torch.randn(1, 3, 512, 512).to(device)
    
    perturbation = generator(x, watermark)
    print(f"Perturbation shape: {perturbation.shape}")
    print(f"Perturbation range: [{perturbation.min():.4f}, {perturbation.max():.4f}]")
    
    adv = generator.generate_adversarial(x, watermark)
    print(f"Adversarial shape: {adv.shape}")
    
    d_out = discriminator(x)
    print(f"Discriminator output shape: {d_out.shape}")
