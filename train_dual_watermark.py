"""
双水印联合训练脚本
==================

同时优化两个目标：
1. 可见水印：让DreamBooth微调后生成的图带有"CUG-BDAIS-328"
2. 频域水印：在VAE latent的频域注入特定模式，用于img2img/ControlNet检测

关键改进：
- 一次嵌入，两种水印
- 端到端联合优化，不互相干扰
- 检测简单：OCR + VAE频域匹配
"""

import os
import sys
import argparse
import time
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
import lpips

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.perturbation_generator import PerturbationGenerator, get_parameter_count


# ==================== 频域水印模块 ====================

class FrequencyWatermark:
    """频域水印模块 - 在VAE latent的FFT域注入特定模式"""
    
    def __init__(self, device, shape=(1, 4, 64, 64), w_channel=3, w_radius=10, seed=328):
        self.device = device
        self.shape = shape
        self.w_channel = w_channel
        self.w_radius = w_radius
        
        torch.manual_seed(seed)
        self.gt_patch, self.watermarking_mask = self._gen_watermark()
        
        # 预计算统计量
        self.mu, self.sigma = self._compute_stats()
        print(f"  频域水印初始化: channel={w_channel}, radius={w_radius}")
        print(f"  统计量: μ={self.mu:.4f}, σ={self.sigma:.4f}")

    def _circle_mask(self, size=64, r=10):
        """生成圆形mask"""
        x0 = y0 = size // 2
        y, x = np.ogrid[:size, :size]
        y = y[::-1]
        return ((x - x0)**2 + (y - y0)**2) <= r**2
    
    def _gen_watermark(self):
        """生成水印模式"""
        gt_init = torch.randn(self.shape, device=self.device)
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        
        # 从外圈到内圈设置水印值
        for r in range(self.w_radius, 0, -1):
            mask = torch.tensor(self._circle_mask(self.shape[-1], r=r)).to(self.device)
            gt_patch[:, self.w_channel, mask] = gt_patch[0, self.w_channel, 0, r].item()
        
        # 生成水印mask
        watermarking_mask = torch.zeros(self.shape, dtype=torch.bool, device=self.device)
        watermarking_mask[:, self.w_channel] = torch.tensor(
            self._circle_mask(self.shape[-1], r=self.w_radius)
        ).to(self.device)
        
        return gt_patch, watermarking_mask
    
    def compute_loss(self, latents):
        """计算频域水印损失 - 让latent的频域接近目标模式"""
        # 扩展batch维度的mask
        batch_size = latents.shape[0]
        mask = self.watermarking_mask.expand(batch_size, -1, -1, -1)
        target = self.gt_patch.expand(batch_size, -1, -1, -1)
        
        # 转换为float32避免ComplexHalf警告
        latents_fft = torch.fft.fftshift(torch.fft.fft2(latents.float()), dim=(-1, -2))
        
        # L1损失：让频域水印区域接近目标
        loss = torch.abs(latents_fft[mask] - target[mask]).mean()
        return loss
    
    def eval_watermark(self, latents):
        """评估水印强度（L1距离，越小越好）"""
        # 只取第一个样本评估
        if latents.shape[0] > 1:
            latents = latents[:1]
        latents_fft = torch.fft.fftshift(torch.fft.fft2(latents.float()), dim=(-1, -2))
        l1_metric = torch.abs(
            latents_fft[self.watermarking_mask] - self.gt_patch[self.watermarking_mask]
        ).mean().item()
        return l1_metric
    
    def _compute_stats(self, n_samples=1000):
        """计算随机噪声的统计量"""
        distances = []
        for _ in range(n_samples):
            rand_latents = torch.randn(self.shape, device=self.device)
            dist = self.eval_watermark(rand_latents)
            distances.append(dist)
        distances = np.array(distances)
        return distances.mean(), distances.var()
    
    def detect(self, latents):
        """检测水印概率
        L1 越小说明越接近水印模式，DP 越高
        """
        l1_metric = self.eval_watermark(latents)
        from scipy.stats import norm
        # L1 < μ 说明有水印，计算单侧概率
        p_value = norm.cdf(l1_metric, self.mu, np.sqrt(self.sigma))
        # 转换为检测概率：p_value 越小，DP 越高
        dp = 1 - p_value
        return min(dp * 2, 1.0)


# ==================== 数据集 ====================

class WatermarkDataset(Dataset):
    """水印对抗样本数据集"""
    def __init__(self, image_dir, watermark_path, transform=None, max_samples=None):
        self.image_dir = image_dir
        self.watermark_path = watermark_path
        self.transform = transform
        
        self.image_paths = []
        for root, dirs, files in os.walk(image_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root, f))
        
        if max_samples:
            self.image_paths = self.image_paths[:max_samples]
        
        self.watermark = Image.open(watermark_path).convert('RGB')
        if self.transform:
            self.watermark = self.transform(self.watermark)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.watermark, os.path.basename(img_path)


def compute_psnr(img1, img2):
    """计算PSNR"""
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return torch.tensor(float('inf'))
    return 10 * torch.log10(1.0 / mse)


# ==================== 训练函数 ====================

def train_dual_watermark(args):
    """双水印联合训练"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"dual_wm_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
    ])
    
    # 数据集
    dataset = WatermarkDataset(
        args.data_dir, 
        args.watermark_path, 
        transform=transform,
        max_samples=args.max_samples
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    print(f"Dataset size: {len(dataset)}")
    
    # 创建扰动生成器
    generator = PerturbationGenerator(eps=args.eps, base_channels=args.base_channels).to(device)
    print(f"Generator params: {get_parameter_count(generator)}")
    
    # 加载LPIPS模型
    print("Loading LPIPS model...")
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    lpips_fn.eval()
    for param in lpips_fn.parameters():
        param.requires_grad = False
    
    # 加载VAE
    vae = None
    if args.vae_path:
        from diffusers.models import AutoencoderKL
        try:
            vae_path = os.path.expanduser(args.vae_path)
            try:
                vae = AutoencoderKL.from_pretrained(vae_path, local_files_only=True)
            except:
                vae = AutoencoderKL.from_pretrained(vae_path, subfolder="vae", local_files_only=True)
            vae = vae.to(device).eval()
            for param in vae.parameters():
                param.requires_grad = False
            print("VAE loaded successfully")
        except Exception as e:
            print(f"Failed to load VAE: {e}")
            vae = None
    
    # 初始化频域水印模块
    print("初始化频域水印模块...")
    freq_wm = FrequencyWatermark(
        device=device,
        w_channel=args.w_channel,
        w_radius=args.w_radius,
        seed=args.wm_seed
    )

    # 优化器
    optim_G = torch.optim.AdamW(generator.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    
    # 学习率调度器
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return epoch / args.warmup_epochs
        return 0.5 * (1 + np.cos((epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs) * np.pi))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim_G, lr_lambda)
    
    # 混合精度
    scaler = GradScaler('cuda') if args.use_amp else None
    
    # 训练循环
    print(f"\n开始训练 {args.epochs} epochs...")
    print(f"损失权重: adv={args.lambda_adv}, freq={args.lambda_freq}, mse={args.lambda_mse}, lpips={args.lambda_lpips}")
    start_time = time.time()
    
    best_psnr = 0
    
    for epoch in range(args.epochs):
        generator.train()
        
        epoch_losses = {
            'adv': 0, 'freq': 0, 'mse': 0, 'lpips': 0, 'total': 0, 'psnr': 0, 'dp': 0
        }
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, (images, watermarks, _) in enumerate(pbar):
            images = images.to(device)
            watermarks = watermarks.to(device)
            
            optim_G.zero_grad()
            
            with autocast('cuda', enabled=args.use_amp):
                # 生成对抗样本
                perturbation = generator(images, watermarks)
                adv_samples = torch.clamp(images + perturbation, 0, 1)
                
                # ========== 损失1：可见水印对抗损失 ==========
                # 让VAE latent接近水印图的latent（DB微调后会生成水印）
                loss_adv = torch.tensor(0.0, device=device)
                if vae is not None:
                    with torch.no_grad():
                        latent_watermark = vae.encode(watermarks * 2 - 1).latent_dist.sample() * 0.18215
                    latent_adv = vae.encode(adv_samples * 2 - 1).latent_dist.sample() * 0.18215
                    loss_adv = F.mse_loss(latent_adv, latent_watermark)
                
                # ========== 损失2：频域水印损失 ==========
                # 让VAE latent的频域有特定模式（用于img2img检测）
                loss_freq = torch.tensor(0.0, device=device)
                if vae is not None:
                    latent_adv_for_freq = vae.encode(adv_samples * 2 - 1).latent_dist.mean * 0.18215
                    loss_freq = freq_wm.compute_loss(latent_adv_for_freq)
                
                # ========== 损失3：视觉质量损失 ==========
                loss_mse = F.mse_loss(adv_samples, images)
                loss_lpips = lpips_fn(adv_samples * 2 - 1, images * 2 - 1).mean()
                
                # 总损失
                loss_total = (
                    args.lambda_adv * loss_adv +
                    args.lambda_freq * loss_freq +
                    args.lambda_mse * loss_mse +
                    args.lambda_lpips * loss_lpips
                )
            
            if scaler:
                scaler.scale(loss_total).backward()
                scaler.unscale_(optim_G)
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
                scaler.step(optim_G)
                scaler.update()
            else:
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
                optim_G.step()
            
            # 计算指标
            with torch.no_grad():
                psnr = compute_psnr(adv_samples, images)
                # 检测频域水印
                if vae is not None:
                    latent_check = vae.encode(adv_samples * 2 - 1).latent_dist.mean * 0.18215
                    dp = freq_wm.detect(latent_check)
                else:
                    dp = 0
            
            # 记录损失
            epoch_losses['adv'] += loss_adv.item()
            epoch_losses['freq'] += loss_freq.item()
            epoch_losses['mse'] += loss_mse.item()
            epoch_losses['lpips'] += loss_lpips.item()
            epoch_losses['total'] += loss_total.item()
            epoch_losses['psnr'] += psnr.item()
            epoch_losses['dp'] += dp
            
            pbar.set_postfix({
                'adv': f"{loss_adv.item():.3f}",
                'freq': f"{loss_freq.item():.1f}",
                'PSNR': f"{psnr.item():.1f}",
                'DP': f"{dp*100:.0f}%"
            })
        
        scheduler.step()

        # 打印epoch统计
        n_batches = len(dataloader)
        avg_psnr = epoch_losses['psnr'] / n_batches
        avg_dp = epoch_losses['dp'] / n_batches
        
        print(f"\nEpoch {epoch+1}/{args.epochs} Summary:")
        print(f"  L_adv (可见水印): {epoch_losses['adv']/n_batches:.4f}")
        print(f"  L_freq (频域水印): {epoch_losses['freq']/n_batches:.2f}")
        print(f"  L_mse: {epoch_losses['mse']/n_batches:.4f}")
        print(f"  L_lpips: {epoch_losses['lpips']/n_batches:.4f}")
        print(f"  PSNR: {avg_psnr:.2f} dB")
        print(f"  频域水印DP: {avg_dp*100:.1f}%")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # 保存样本
        if (epoch + 1) % args.save_freq == 0 or epoch == 0:
            generator.eval()
            with torch.no_grad():
                sample_img = images[:1]
                sample_wm = watermarks[:1]
                sample_pert = generator(sample_img, sample_wm)
                sample_adv = torch.clamp(sample_img + sample_pert, 0, 1)
                
                # 放大扰动以便可视化
                pert_vis = (sample_pert * 10 + 0.5).clamp(0, 1)
                
                comparison = torch.cat([sample_img, sample_wm, sample_adv, pert_vis], dim=0)
                save_image(comparison, 
                          os.path.join(output_dir, "samples", f"epoch_{epoch+1}.png"),
                          nrow=4)
            generator.train()
        
        # 保存最佳模型
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save({
                'epoch': epoch + 1,
                'generator': generator.state_dict(),
                'psnr': avg_psnr,
                'dp': avg_dp,
                'freq_wm': {
                    'gt_patch': freq_wm.gt_patch,
                    'watermarking_mask': freq_wm.watermarking_mask,
                    'mu': freq_wm.mu,
                    'sigma': freq_wm.sigma,
                    'w_channel': freq_wm.w_channel,
                    'w_radius': freq_wm.w_radius,
                },
                'config': vars(args)
            }, os.path.join(output_dir, "best_model.pt"))
        
        # 保存检查点
        if (epoch + 1) % args.checkpoint_freq == 0:
            torch.save({
                'epoch': epoch + 1,
                'generator': generator.state_dict(),
                'optimizer': optim_G.state_dict(),
                'psnr': avg_psnr,
                'dp': avg_dp,
            }, os.path.join(output_dir, "checkpoints", f"checkpoint_{epoch+1}.pt"))
    
    # 保存最终模型
    torch.save({
        'generator': generator.state_dict(),
        'freq_wm': {
            'gt_patch': freq_wm.gt_patch,
            'watermarking_mask': freq_wm.watermarking_mask,
            'mu': freq_wm.mu,
            'sigma': freq_wm.sigma,
            'w_channel': freq_wm.w_channel,
            'w_radius': freq_wm.w_radius,
        },
        'config': vars(args),
        'final_psnr': avg_psnr,
        'final_dp': avg_dp
    }, os.path.join(output_dir, "final_model.pt"))
    
    total_time = time.time() - start_time
    print(f"\n训练完成，耗时 {total_time/60:.2f} 分钟")
    print(f"最佳 PSNR: {best_psnr:.2f} dB")
    print(f"模型保存至: {output_dir}")
    
    return generator, freq_wm, output_dir


def main():
    parser = argparse.ArgumentParser(description='双水印联合训练')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, required=True, help='训练图像目录')
    parser.add_argument('--watermark_path', type=str, required=True, help='可见水印图像路径')
    parser.add_argument('--output_dir', type=str, default='./output_dual_wm', help='输出目录')
    parser.add_argument('--vae_path', type=str, default='../sd1.5', help='VAE模型路径')
    
    # 模型参数
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--base_channels', type=int, default=32)
    parser.add_argument('--eps', type=float, default=8/255, help='扰动范围')
    
    # 频域水印参数
    parser.add_argument('--w_channel', type=int, default=3, help='水印通道')
    parser.add_argument('--w_radius', type=int, default=10, help='水印半径')
    parser.add_argument('--wm_seed', type=int, default=328, help='水印随机种子')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--max_samples', type=int, default=1000, help='最大训练样本数')
    parser.add_argument('--num_workers', type=int, default=4)
    
    # 损失权重
    parser.add_argument('--lambda_adv', type=float, default=1.0, help='可见水印对抗损失')
    parser.add_argument('--lambda_freq', type=float, default=0.1, help='频域水印损失')
    parser.add_argument('--lambda_mse', type=float, default=100.0, help='MSE损失')
    parser.add_argument('--lambda_lpips', type=float, default=10.0, help='LPIPS损失')
    
    # 其他
    parser.add_argument('--use_amp', action='store_true', help='使用混合精度')
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--checkpoint_freq', type=int, default=20)
    
    args = parser.parse_args()
    train_dual_watermark(args)


if __name__ == '__main__':
    main()
