"""
DreamBooth训练 - 测试对抗样本效果
"""

import os
import argparse
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


class DreamBoothDataset(Dataset):
    def __init__(self, data_dir, tokenizer, size=512, prompt="a satellite image of sks scene"):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.size = size
        self.prompt = prompt
        
        self.images = []
        print(f"扫描目录: {data_dir}")
        for root, dirs, files in os.walk(data_dir):
            print(f"  子目录: {root}, 文件数: {len(files)}")
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(root, f))
        print(f"找到 {len(self.images)} 张图片")
        
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        self.input_ids = tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt"
        ).input_ids[0]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        img = self.transform(img)
        return {"pixel_values": img, "input_ids": self.input_ids}


def train_dreambooth(data_dir, output_dir, sd_path, num_steps=400, lr=5e-6, batch_size=1):
    device = torch.device("cuda")
    
    print(f"加载模型: {sd_path}")
    tokenizer = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer", local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder", local_files_only=True).to(device)
    vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae", local_files_only=True).to(device)
    unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet", local_files_only=True).to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(sd_path, subfolder="scheduler", local_files_only=True)
    
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()
    
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr)
    
    dataset = DreamBoothDataset(data_dir, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"数据集: {len(dataset)} 张图像")
    print(f"训练 {num_steps} 步...")
    
    global_step = 0
    progress_bar = tqdm(total=num_steps)
    
    while global_step < num_steps:
        for batch in dataloader:
            if global_step >= num_steps:
                break
            
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
                encoder_hidden_states = text_encoder(input_ids)[0]
            
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())
    
    progress_bar.close()
    
    # 保存模型
    print(f"保存模型到: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    pipeline = StableDiffusionPipeline.from_pretrained(
        sd_path,
        unet=unet,
        local_files_only=True,
        safety_checker=None
    )
    pipeline.save_pretrained(output_dir)
    print("完成!")
    
    return output_dir


def generate_images_single_scene(model_dir, output_dir, scene, num_images=6, seed=42):
    """为单个场景生成图像"""
    print(f"加载模型: {model_dir}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")
    
    os.makedirs(output_dir, exist_ok=True)
    
    prompt = f"a satellite image of sks {scene.lower()}"
    print(f"  生成 {num_images} 张: {prompt}")
    
    for j in range(num_images):
        gen = torch.Generator(device="cuda").manual_seed(seed + j)
        image = pipe(prompt, num_inference_steps=50, generator=gen).images[0]
        image.save(os.path.join(output_dir, f"{scene}_{j+1:03d}.png"))
    
    print(f"保存至: {output_dir}")
    
    # 释放显存
    del pipe
    torch.cuda.empty_cache()


def generate_images(model_dir, output_dir, num_images_per_scene=6, seed=42):
    print(f"加载模型: {model_dir}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 15个场景
    SCENES = [
        "Airport", "Bridge", "Center", "Church", "Farmland",
        "Forest", "Meadow", "Mountain", "Parking", "Port",
        "River", "School", "Square", "Stadium", "StorageTanks"
    ]
    
    print(f"生成图像 ({len(SCENES)} 场景, 每场景 {num_images_per_scene} 张)...")
    
    for scene in SCENES:
        # 创建场景目录
        scene_dir = os.path.join(output_dir, scene)
        os.makedirs(scene_dir, exist_ok=True)
        
        prompt = f"a satellite image of sks {scene.lower()}"
        print(f"  场景: {scene} - {prompt}")
        
        for j in range(num_images_per_scene):
            gen = torch.Generator(device="cuda").manual_seed(seed + j)
            image = pipe(prompt, num_inference_steps=50, generator=gen).images[0]
            # 命名格式：{场景名}_{序号:03d}.png
            image.save(os.path.join(scene_dir, f"{scene}_{j+1:03d}.png"))
    
    print(f"保存至: {output_dir}")


def main(args):
    if args.per_scene:
        # 按场景分别训练
        scenes = [d for d in os.listdir(args.data_dir) 
                  if os.path.isdir(os.path.join(args.data_dir, d))]
        scenes = sorted(scenes)
        
        # 从指定场景开始
        if args.start_from:
            if args.start_from in scenes:
                start_idx = scenes.index(args.start_from)
                scenes = scenes[start_idx:]
                print(f"从 {args.start_from} 继续，剩余 {len(scenes)} 个场景")
            else:
                print(f"错误: 场景 {args.start_from} 不存在")
                return
        
        print(f"按场景训练模式: {len(scenes)} 个场景")
        print(f"场景列表: {scenes}")
        
        for scene in scenes:
            print(f"\n{'='*60}")
            print(f"训练场景: {scene}")
            print(f"{'='*60}")
            
            scene_data_dir = os.path.join(args.data_dir, scene)
            scene_output_dir = os.path.join(args.output_dir, scene)
            os.makedirs(scene_output_dir, exist_ok=True)
            
            # 训练
            model_dir = os.path.join(scene_output_dir, "model")
            train_dreambooth(scene_data_dir, model_dir, args.sd_path, args.num_steps, args.lr)
            
            # 生成 - 只生成当前场景
            gen_dir = os.path.join(scene_output_dir, "generated")
            generate_images_single_scene(model_dir, gen_dir, scene, args.num_images, args.seed)
            
            # 删除模型节省空间
            if args.delete_model:
                import shutil
                print(f"删除模型目录: {model_dir}")
                shutil.rmtree(model_dir, ignore_errors=True)
        
        print(f"\n全部完成！共训练 {len(scenes)} 个场景")
    else:
        # 原有逻辑：所有图片一起训练
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.join(args.output_dir, f"dreambooth_{timestamp}")
        os.makedirs(base_dir, exist_ok=True)
        
        # 训练
        model_dir = os.path.join(base_dir, "model")
        train_dreambooth(args.data_dir, model_dir, args.sd_path, args.num_steps, args.lr)
        
        # 生成
        gen_dir = os.path.join(base_dir, "generated")
        generate_images(model_dir, gen_dir, args.num_images, args.seed)
        
        print(f"\n完成！检查 {gen_dir} 中的图像是否出现'CUG-BDAIS-328'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='训练数据目录')
    parser.add_argument('--sd_path', type=str, default='../sd1.5')
    parser.add_argument('--output_dir', type=str, default='./output_dreambooth')
    parser.add_argument('--num_steps', type=int, default=400)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--num_images', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--per_scene', action='store_true', help='按场景分别训练')
    parser.add_argument('--start_from', type=str, default=None, help='从指定场景开始继续训练')
    parser.add_argument('--delete_model', action='store_true', help='生成后删除模型节省空间')
    
    args = parser.parse_args()
    main(args)
