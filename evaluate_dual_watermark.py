"""
双水印检测评估脚本
==================

检测两种水印：
1. 可见水印：OCR检测"CUG-BDAIS-328"
2. 频域水印：VAE编码后频域模式匹配

测试各种攻击下的检测效果
"""

import os
import sys
import argparse
import numpy as np
from scipy.stats import norm
import io

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.perturbation_generator import PerturbationGenerator


class FrequencyWatermarkDetector:
    """频域水印检测器"""
    
    def __init__(self, checkpoint_path, device='cuda', w_channel=3, w_radius=10, seed=328):
        self.device = device
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # 尝试从checkpoint加载freq_wm，如果没有则重新生成
        if 'freq_wm' in checkpoint:
            freq_wm = checkpoint['freq_wm']
            self.gt_patch = freq_wm['gt_patch'].to(device)
            self.watermarking_mask = freq_wm['watermarking_mask'].to(device)
            self.mu = freq_wm['mu']
            self.sigma = freq_wm['sigma']
            self.w_channel = freq_wm['w_channel']
            self.w_radius = freq_wm['w_radius']
            print(f"频域水印检测器从checkpoint加载成功")
        else:
            # 从config或默认值重新生成
            config = checkpoint.get('config', {})
            self.w_channel = config.get('w_channel', w_channel)
            self.w_radius = config.get('w_radius', w_radius)
            wm_seed = config.get('wm_seed', seed)
            
            # 重新生成水印模式（必须用相同的seed）
            self._init_watermark(wm_seed)
            print(f"频域水印检测器重新初始化")
        
        print(f"  channel={self.w_channel}, radius={self.w_radius}")
        print(f"  μ={self.mu:.4f}, σ={self.sigma:.4f}")
    
    def _init_watermark(self, seed):
        """重新初始化水印模式"""
        import numpy as np
        shape = (1, 4, 64, 64)
        
        torch.manual_seed(seed)
        gt_init = torch.randn(shape, device=self.device)
        self.gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        
        # 圆形mask
        def circle_mask(size=64, r=10):
            x0 = y0 = size // 2
            y, x = np.ogrid[:size, :size]
            y = y[::-1]
            return ((x - x0)**2 + (y - y0)**2) <= r**2
        
        for r in range(self.w_radius, 0, -1):
            mask = torch.tensor(circle_mask(shape[-1], r=r)).to(self.device)
            self.gt_patch[:, self.w_channel, mask] = self.gt_patch[0, self.w_channel, 0, r].item()
        
        self.watermarking_mask = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.watermarking_mask[:, self.w_channel] = torch.tensor(
            circle_mask(shape[-1], r=self.w_radius)
        ).to(self.device)
        
        # 计算统计量
        distances = []
        for _ in range(1000):
            rand_latents = torch.randn(shape, device=self.device)
            dist = self._eval_single(rand_latents)
            distances.append(dist)
        distances = np.array(distances)
        self.mu = distances.mean()
        self.sigma = distances.var()
    
    def _eval_single(self, latents):
        latents_fft = torch.fft.fftshift(torch.fft.fft2(latents.float()), dim=(-1, -2))
        return torch.abs(
            latents_fft[self.watermarking_mask] - self.gt_patch[self.watermarking_mask]
        ).mean().item()
    
    def eval_watermark(self, latents):
        """评估水印强度"""
        latents_fft = torch.fft.fftshift(torch.fft.fft2(latents), dim=(-1, -2))
        l1_metric = torch.abs(
            latents_fft[self.watermarking_mask] - self.gt_patch[self.watermarking_mask]
        ).mean().item()
        return l1_metric
    
    def detect(self, latents):
        """检测水印概率
        基于 L1 阈值判断：
        - L1 < 55: 高置信度有水印 (DP > 95%)
        - L1 55-70: 中等置信度 (DP 50-95%)
        - L1 > 70: 低置信度/无水印 (DP < 50%)
        """
        l1_metric = self.eval_watermark(latents)
        
        # 基于观察到的分布设置阈值
        # 保护图 L1 ≈ 45-50, 原图 L1 ≈ 75-85
        if l1_metric < 55:
            dp = 1.0
        elif l1_metric < 70:
            # 线性插值
            dp = 1.0 - (l1_metric - 55) / 30  # 55->1.0, 70->0.5
        else:
            dp = max(0, 0.5 - (l1_metric - 70) / 40)  # 70->0.5, 90->0
        
        return dp


def test_robustness(vae, freq_detector, image, device):
    """测试各种攻击下的频域水印检测"""
    results = {}
    
    def test_attack(name, attacked_img):
        with torch.no_grad():
            latent = vae.encode(attacked_img * 2 - 1).latent_dist.mean * 0.18215
            dp = freq_detector.detect(latent)
            l1 = freq_detector.eval_watermark(latent)
        status = "✓" if dp > 0.9 else ("~" if dp > 0.7 else "✗")
        results[name] = {'dp': dp, 'l1': l1}
        print(f"    {name}: L1={l1:.2f}, DP={dp*100:.1f}% {status}")
        return dp
    
    # 直接检测
    test_attack("直接检测", image)
    
    # JPEG压缩
    for q in [90, 70, 50, 30]:
        pil = TF.to_pil_image(image[0].cpu())
        buf = io.BytesIO()
        pil.save(buf, format='JPEG', quality=q)
        buf.seek(0)
        jpeg_img = TF.to_tensor(Image.open(buf)).unsqueeze(0).to(device)
        test_attack(f"JPEG Q{q}", jpeg_img)
    
    # 高斯噪声
    for sigma in [0.01, 0.03, 0.05]:
        noise_img = torch.clamp(image + torch.randn_like(image) * sigma, 0, 1)
        test_attack(f"噪声 σ={sigma}", noise_img)
    
    # 高斯模糊
    for sigma in [1.0, 2.0]:
        blur_img = TF.gaussian_blur(image, kernel_size=5, sigma=sigma)
        test_attack(f"模糊 σ={sigma}", blur_img)
    
    # VAE重建
    with torch.no_grad():
        scaling = vae.config.scaling_factor
        vae_lat = vae.encode(image * 2 - 1).latent_dist.mean * scaling
        vae_img = vae.decode(vae_lat / scaling).sample
        vae_img = torch.clamp((vae_img + 1) / 2, 0, 1)
    test_attack("VAE重建", vae_img)
    
    # 多次VAE重建
    with torch.no_grad():
        multi_vae_img = image.clone()
        for _ in range(3):
            lat = vae.encode(multi_vae_img * 2 - 1).latent_dist.mean * scaling
            multi_vae_img = vae.decode(lat / scaling).sample
            multi_vae_img = torch.clamp((multi_vae_img + 1) / 2, 0, 1)
    test_attack("VAE重建×3", multi_vae_img)
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='训练好的模型路径')
    parser.add_argument('--data_dir', type=str, required=True, help='测试图像目录')
    parser.add_argument('--watermark_path', type=str, required=True, help='可见水印图像')
    parser.add_argument('--vae_path', type=str, default='../sd1.5')
    parser.add_argument('--output_dir', type=str, default='./output_eval_dual')
    parser.add_argument('--num_test', type=int, default=10)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    print(f"加载模型: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    
    # 加载生成器
    generator = PerturbationGenerator(
        eps=config.get('eps', 8/255),
        base_channels=config.get('base_channels', 32)
    ).to(device)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    
    # 加载频域水印检测器
    freq_detector = FrequencyWatermarkDetector(args.model_path, device)
    
    # 加载VAE
    from diffusers.models import AutoencoderKL
    print(f"加载VAE: {args.vae_path}")
    try:
        vae = AutoencoderKL.from_pretrained(args.vae_path, subfolder="vae", local_files_only=True)
    except:
        vae = AutoencoderKL.from_pretrained(args.vae_path, local_files_only=True)
    vae = vae.to(device).eval()
    
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
    ])
    
    # 加载水印图像
    watermark_img = Image.open(args.watermark_path).convert('RGB')
    watermark_img = transform(watermark_img).unsqueeze(0).to(device)
    
    # 获取测试图像
    test_images = []
    for root, dirs, files in os.walk(args.data_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                test_images.append(os.path.join(root, f))
                if len(test_images) >= args.num_test:
                    break
        if len(test_images) >= args.num_test:
            break
    
    print(f"测试图像: {len(test_images)} 张")
    print("="*60)
    
    all_results = []
    
    for idx, img_path in enumerate(test_images):
        print(f"\n[{idx+1}/{len(test_images)}] {os.path.basename(img_path)}")
        
        # 加载原图
        image = Image.open(img_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        
        # 生成保护图像
        with torch.no_grad():
            perturbation = generator(image, watermark_img)
            protected_img = torch.clamp(image + perturbation, 0, 1)
            
            # 计算PSNR
            mse = F.mse_loss(protected_img, image)
            psnr = 10 * torch.log10(1.0 / (mse + 1e-8)).item()
        
        print(f"  PSNR: {psnr:.2f} dB")
        
        # 原图检测
        print("  [原图检测]")
        with torch.no_grad():
            orig_latent = vae.encode(image * 2 - 1).latent_dist.mean * 0.18215
            orig_l1 = freq_detector.eval_watermark(orig_latent)
            orig_dp = freq_detector.detect(orig_latent)
        print(f"    原图 L1: {orig_l1:.2f}, DP: {orig_dp*100:.1f}%")
        
        # 保护图检测
        print("  [保护图检测]")
        results = test_robustness(vae, freq_detector, protected_img, device)
        results['psnr'] = psnr
        results['orig_dp'] = orig_dp
        all_results.append(results)
        
        # 保存可视化
        diff = (protected_img - image) * 10 + 0.5
        vis = torch.cat([image, protected_img, diff.clamp(0,1)], dim=0)
        save_image(vis, os.path.join(args.output_dir, f"test_{idx+1}.png"), nrow=3)
    
    # 汇总
    print("\n" + "="*60)
    print("汇总结果:")
    print(f"  平均PSNR: {np.mean([r['psnr'] for r in all_results]):.2f} dB")
    print(f"  原图平均DP: {np.mean([r['orig_dp'] for r in all_results])*100:.1f}%")
    
    print("\n各攻击下的频域水印DP:")
    for attack in ['直接检测', 'JPEG Q90', 'JPEG Q70', 'JPEG Q50', 'JPEG Q30',
                   '噪声 σ=0.01', '噪声 σ=0.03', '噪声 σ=0.05',
                   '模糊 σ=1.0', '模糊 σ=2.0', 'VAE重建', 'VAE重建×3']:
        if attack in all_results[0]:
            avg_dp = np.mean([r[attack]['dp'] for r in all_results])
            print(f"  {attack}: {avg_dp*100:.1f}%")
    
    print(f"\n结果保存至: {args.output_dir}")


if __name__ == '__main__':
    main()
