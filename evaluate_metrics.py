#!/usr/bin/env python3
"""
评估脚本：计算 FID、OA、LPIPS 指标
用于表2基线方法对比
"""

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms, models
from collections import defaultdict
from scipy import linalg
import lpips


class SceneClassifier(nn.Module):
    """场景分类器"""
    def __init__(self, num_classes=30, pretrained_path=None):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(2048, num_classes)
        self.classes = None
        
        if pretrained_path and os.path.exists(pretrained_path):
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                self.backbone.load_state_dict(checkpoint['model_state_dict'])
                self.classes = checkpoint.get('classes', None)
                print(f"Loaded classifier (Acc: {checkpoint.get('best_acc', 'N/A')}%)")
    
    def forward(self, x):
        return self.backbone(x)


class FeatureExtractor(nn.Module):
    """特征提取器（用于SSM和FID）"""
    def __init__(self):
        super().__init__()
        self.model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Identity()
        self.model.eval()
    
    def forward(self, x):
        with torch.no_grad():
            return self.model(x)


def calculate_fid(real_features, gen_features):
    """计算FID"""
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)
    
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def calculate_ssm(orig_features, gen_features):
    """计算场景语义匹配度（余弦相似度）"""
    orig_norm = orig_features / np.linalg.norm(orig_features, axis=1, keepdims=True)
    gen_norm = gen_features / np.linalg.norm(gen_features, axis=1, keepdims=True)
    similarities = np.sum(orig_norm * gen_norm, axis=1)
    return float(np.mean(similarities))


def calculate_lpips(orig_dir, gen_dir, device, max_pairs=100):
    """计算LPIPS分数"""
    loss_fn = lpips.LPIPS(net='alex').to(device)
    loss_fn.eval()
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    scores = []
    
    # 遍历场景
    for scene in sorted(os.listdir(gen_dir)):
        gen_scene_dir = os.path.join(gen_dir, scene)
        orig_scene_dir = os.path.join(orig_dir, scene)
        
        if not os.path.isdir(gen_scene_dir) or not os.path.exists(orig_scene_dir):
            continue
        
        # 支持 generated 子目录
        if os.path.exists(os.path.join(gen_scene_dir, 'generated')):
            gen_img_dir = os.path.join(gen_scene_dir, 'generated')
        else:
            gen_img_dir = gen_scene_dir
        
        gen_images = [f for f in os.listdir(gen_img_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        orig_images = [f for f in os.listdir(orig_scene_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not gen_images or not orig_images:
            continue
        
        # 随机配对计算
        n_pairs = min(len(gen_images), len(orig_images), 10)
        for i in range(n_pairs):
            gen_path = os.path.join(gen_img_dir, gen_images[i % len(gen_images)])
            orig_path = os.path.join(orig_scene_dir, orig_images[i % len(orig_images)])
            
            gen_img = transform(Image.open(gen_path).convert('RGB')).unsqueeze(0).to(device)
            orig_img = transform(Image.open(orig_path).convert('RGB')).unsqueeze(0).to(device)
            
            with torch.no_grad():
                score = loss_fn(gen_img, orig_img)
            scores.append(score.item())
    
    return float(np.mean(scores)) if scores else 0.0


def main():
    parser = argparse.ArgumentParser(description='评估生成图像指标')
    parser.add_argument('--gen_dir', type=str, required=True, help='生成图像目录')
    parser.add_argument('--orig_dir', type=str, required=True, help='原始图像目录')
    parser.add_argument('--classifier_path', type=str, default='./models/aid_classifier.pth')
    parser.add_argument('--num_classes', type=int, default=30)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 加载分类器
    classifier = SceneClassifier(args.num_classes, args.classifier_path).to(device)
    classifier.eval()
    
    # 加载特征提取器
    feature_extractor = FeatureExtractor().to(device)
    
    # 获取类别映射
    classes = classifier.classes
    if classes:
        class_to_idx = {c: i for i, c in enumerate(classes)}
    else:
        class_to_idx = {}
    
    # 图像变换
    classify_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    feature_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 统计
    all_orig_features = []
    all_gen_features = []
    total_correct = 0
    total_count = 0
    
    print(f"\nEvaluating: {args.gen_dir}")
    print(f"Original: {args.orig_dir}")
    print()
    
    # 遍历场景
    for scene in sorted(os.listdir(args.gen_dir)):
        gen_scene_dir = os.path.join(args.gen_dir, scene)
        orig_scene_dir = os.path.join(args.orig_dir, scene)
        
        if not os.path.isdir(gen_scene_dir):
            continue
        
        if not os.path.exists(orig_scene_dir):
            print(f"SKIP: {scene} (not found in orig_dir)")
            continue
        
        # 获取真实标签
        if scene in class_to_idx:
            true_label = class_to_idx[scene]
        else:
            print(f"WARNING: {scene} not in classifier classes")
            true_label = -1
        
        correct = 0
        count = 0
        predictions = defaultdict(int)
        scene_orig_features = []
        scene_gen_features = []
        
        # 遍历该场景的图像（支持 generated 子目录）
        if os.path.exists(os.path.join(gen_scene_dir, 'generated')):
            gen_img_dir = os.path.join(gen_scene_dir, 'generated')
        else:
            gen_img_dir = gen_scene_dir
        
        gen_images = sorted([f for f in os.listdir(gen_img_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        for img_name in gen_images:
            gen_path = os.path.join(gen_img_dir, img_name)
            orig_path = os.path.join(orig_scene_dir, img_name)
            
            if not os.path.exists(orig_path):
                # 尝试匹配
                base_name = os.path.splitext(img_name)[0]
                for ext in ['.png', '.jpg', '.jpeg']:
                    alt_path = os.path.join(orig_scene_dir, base_name + ext)
                    if os.path.exists(alt_path):
                        orig_path = alt_path
                        break
            
            if not os.path.exists(orig_path):
                continue
            
            # 加载图像
            gen_img = Image.open(gen_path).convert('RGB')
            orig_img = Image.open(orig_path).convert('RGB')
            
            # 分类
            gen_tensor = classify_transform(gen_img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = classifier(gen_tensor)
                pred = output.argmax(1).item()
            
            pred_class = classes[pred] if classes else str(pred)
            predictions[pred_class] += 1
            
            if pred == true_label:
                correct += 1
            count += 1
            
            # 提取特征
            gen_feat_tensor = feature_transform(gen_img).unsqueeze(0).to(device)
            orig_feat_tensor = feature_transform(orig_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                gen_feat = feature_extractor(gen_feat_tensor).cpu().numpy()
                orig_feat = feature_extractor(orig_feat_tensor).cpu().numpy()
            
            scene_gen_features.append(gen_feat[0])
            scene_orig_features.append(orig_feat[0])
        
        if count > 0:
            acc = correct / count * 100
            total_correct += correct
            total_count += count
            
            all_orig_features.extend(scene_orig_features)
            all_gen_features.extend(scene_gen_features)
            
            print(f"{scene}: {correct}/{count} ({acc:.1f}%) -> {dict(predictions)}")
    
    print()
    print("=" * 60)
    
    # 计算总体指标
    if total_count > 0:
        overall_acc = total_correct / total_count * 100
        oa = overall_acc / 100  # 转换为0-1范围
        print(f"Overall Accuracy (OA): {overall_acc:.1f}%")
    
    if len(all_orig_features) > 1:
        all_orig_features = np.array(all_orig_features)
        all_gen_features = np.array(all_gen_features)
        
        # FID
        fid = calculate_fid(all_orig_features, all_gen_features)
        print(f"FID: {fid:.1f}")
        
        # SSM
        ssm = calculate_ssm(all_orig_features, all_gen_features)
        print(f"SSM: {ssm:.3f}")
    
    # LPIPS
    print("\nCalculating LPIPS...")
    lpips_score = calculate_lpips(args.orig_dir, args.gen_dir, device)
    print(f"LPIPS: {lpips_score:.4f}")
    
    print("=" * 60)
    
    # 输出表格格式
    print("\n表格数据 (表2格式):")
    if 'fid' in dir():
        print(f"FID↑: {fid:.0f}")
    if 'oa' in dir():
        print(f"OA↓: {oa:.2f}")
    if 'lpips_score' in dir():
        print(f"LPIPS↑: {lpips_score:.2f}")


if __name__ == '__main__':
    main()
