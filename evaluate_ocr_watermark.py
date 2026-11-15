"""
OCR-based Visible Watermark Evaluation Script
评估生成图像中可见水印"CUG-BDAIS-328"的识别率

Usage:
    python evaluate_ocr_watermark.py --image_dir <path_to_images>
    python evaluate_ocr_watermark.py --image_dir <path_to_images> --target_text "CUG-BDAIS-328"
"""

import os
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from collections import defaultdict

try:
    import easyocr
    USE_EASYOCR = True
except ImportError:
    USE_EASYOCR = False
    print("Warning: easyocr not installed. Install with: pip install easyocr")

try:
    import pytesseract
    USE_TESSERACT = True
except ImportError:
    USE_TESSERACT = False
    print("Warning: pytesseract not installed. Install with: pip install pytesseract")


def normalize_text(text):
    """Normalize text for comparison (lowercase, remove spaces/special chars)"""
    return ''.join(c.lower() for c in text if c.isalnum())


def check_watermark_match(detected_texts, target_text="CUG-BDAIS-328", threshold=0.6):
    """
    Check if target watermark text is detected in OCR results
    
    Args:
        detected_texts: list of detected text strings
        target_text: target watermark string to match
        threshold: minimum similarity ratio for fuzzy matching
    
    Returns:
        (is_matched, best_match, confidence)
    """
    target_normalized = normalize_text(target_text)
    
    for text in detected_texts:
        text_normalized = normalize_text(text)
        
        # Exact match
        if target_normalized in text_normalized or text_normalized in target_normalized:
            return True, text, 1.0
        
        # Fuzzy match using character overlap
        if len(text_normalized) > 0 and len(target_normalized) > 0:
            # Check if key parts are present
            key_parts = ['cug', 'bdais', '328']
            matches = sum(1 for part in key_parts if part in text_normalized)
            if matches >= 2:  # At least 2 out of 3 key parts
                return True, text, matches / 3.0
    
    return False, None, 0.0


def ocr_with_easyocr(image_path, reader):
    """Perform OCR using EasyOCR"""
    results = reader.readtext(str(image_path))
    texts = [result[1] for result in results]
    confidences = [result[2] for result in results]
    return texts, confidences


def ocr_with_tesseract(image_path):
    """Perform OCR using Tesseract"""
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    # Split into lines and filter empty
    texts = [line.strip() for line in text.split('\n') if line.strip()]
    return texts, [1.0] * len(texts)


def evaluate_single_image(image_path, reader=None, target_text="CUG-BDAIS-328"):
    """Evaluate watermark detection for a single image"""
    results = {
        'path': str(image_path),
        'detected': False,
        'matched_text': None,
        'confidence': 0.0,
        'all_texts': []
    }
    
    try:
        if USE_EASYOCR and reader is not None:
            texts, confidences = ocr_with_easyocr(image_path, reader)
        elif USE_TESSERACT:
            texts, confidences = ocr_with_tesseract(image_path)
        else:
            print("No OCR engine available!")
            return results
        
        results['all_texts'] = texts
        is_matched, matched_text, confidence = check_watermark_match(texts, target_text)
        results['detected'] = is_matched
        results['matched_text'] = matched_text
        results['confidence'] = confidence
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
    
    return results


def evaluate_directory(image_dir, target_text="CUG-BDAIS-328", extensions=['.png', '.jpg', '.jpeg']):
    """
    Evaluate all images in a directory
    
    Args:
        image_dir: path to directory containing images
        target_text: watermark text to detect
        extensions: list of valid image extensions
    
    Returns:
        dict with evaluation results
    """
    image_dir = Path(image_dir)
    
    # Initialize OCR reader
    reader = None
    if USE_EASYOCR:
        print("Initializing EasyOCR reader...")
        reader = easyocr.Reader(['en'], gpu=True)
    
    # Find all images
    image_files = []
    for ext in extensions:
        image_files.extend(image_dir.glob(f'*{ext}'))
        image_files.extend(image_dir.glob(f'*{ext.upper()}'))
    
    print(f"Found {len(image_files)} images in {image_dir}")
    
    # Evaluate each image
    all_results = []
    detected_count = 0
    
    for i, image_path in enumerate(sorted(image_files)):
        print(f"Processing [{i+1}/{len(image_files)}]: {image_path.name}")
        result = evaluate_single_image(image_path, reader, target_text)
        all_results.append(result)
        
        if result['detected']:
            detected_count += 1
            print(f"  ✓ Watermark detected: '{result['matched_text']}' (conf: {result['confidence']:.2f})")
        else:
            print(f"  ✗ Watermark not detected. Found texts: {result['all_texts'][:3]}")
    
    # Calculate statistics
    total = len(image_files)
    detection_rate = detected_count / total * 100 if total > 0 else 0
    
    summary = {
        'total_images': total,
        'detected_count': detected_count,
        'detection_rate': detection_rate,
        'target_text': target_text,
        'results': all_results
    }
    
    return summary


def evaluate_directory_recursive(image_dir, reader=None, target_text="CUG-BDAIS-328", extensions=['.png', '.jpg', '.jpeg']):
    """
    Recursively evaluate all images in a directory and its subdirectories
    (e.g., robustness_test/db_none/Center/*.png)
    """
    image_dir = Path(image_dir)
    
    # Find all images recursively
    image_files = []
    for ext in extensions:
        image_files.extend(image_dir.rglob(f'*{ext}'))
        image_files.extend(image_dir.rglob(f'*{ext.upper()}'))
    
    print(f"Found {len(image_files)} images in {image_dir} (recursive)")
    
    all_results = []
    detected_count = 0
    
    for i, image_path in enumerate(sorted(image_files)):
        rel_path = image_path.relative_to(image_dir)
        print(f"  [{i+1}/{len(image_files)}] {rel_path}", end=" ")
        result = evaluate_single_image(image_path, reader, target_text)
        all_results.append(result)
        
        if result['detected']:
            detected_count += 1
            matched = result['matched_text'][:20] if result['matched_text'] else ""
            print(f"✓ '{matched}...'")
        else:
            print(f"✗")
    
    total = len(image_files)
    detection_rate = detected_count / total * 100 if total > 0 else 0
    
    return {
        'total_images': total,
        'detected_count': detected_count,
        'detection_rate': detection_rate,
        'target_text': target_text,
        'results': all_results
    }


def evaluate_by_attack_type(base_dir, attack_dirs, target_text="CUG-BDAIS-328"):
    """
    Evaluate watermark detection across different attack types
    Supports nested directory structure: base_dir/attack_type/scene/*.png
    
    Args:
        base_dir: base directory containing attack subdirectories
        attack_dirs: dict mapping attack names to subdirectory names
        target_text: watermark text to detect
    
    Returns:
        dict with results per attack type
    """
    base_dir = Path(base_dir)
    results_by_attack = {}
    
    # Initialize OCR reader once for all evaluations
    reader = None
    if USE_EASYOCR:
        print("Initializing EasyOCR reader...")
        reader = easyocr.Reader(['en'], gpu=True)
    
    for attack_name, subdir in attack_dirs.items():
        attack_path = base_dir / subdir
        if attack_path.exists():
            print(f"\n{'='*60}")
            print(f"Evaluating: {attack_name} ({subdir})")
            print(f"{'='*60}")
            
            result = evaluate_directory_recursive(attack_path, reader, target_text)
            results_by_attack[attack_name] = result
            
            print(f">>> {attack_name}: {result['detection_rate']:.1f}% ({result['detected_count']}/{result['total_images']})")
        else:
            print(f"Warning: Directory not found: {attack_path}")
    
    return results_by_attack


def print_summary_table(results_by_attack):
    """Print summary table of detection rates"""
    print("\n" + "="*60)
    print("SUMMARY: Visible Watermark Detection Rate (OCR)")
    print("="*60)
    print(f"{'Attack Type':<25} {'Total':<10} {'Detected':<10} {'Rate':<10}")
    print("-"*60)
    
    for attack_name, result in results_by_attack.items():
        print(f"{attack_name:<25} {result['total_images']:<10} {result['detected_count']:<10} {result['detection_rate']:.1f}%")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='OCR-based Visible Watermark Evaluation')
    parser.add_argument('--image_dir', type=str, help='Directory containing images to evaluate')
    parser.add_argument('--base_dir', type=str, help='Base directory for multi-attack evaluation')
    parser.add_argument('--target_text', type=str, default='CUG-BDAIS-328', 
                        help='Target watermark text to detect')
    parser.add_argument('--output', type=str, default='ocr_results.txt',
                        help='Output file for results')
    parser.add_argument('--mode', type=str, default='db', choices=['db', 'attacked', 'all'],
                        help='Evaluation mode: db (DreamBooth generated), attacked (preprocessed), all')
    
    args = parser.parse_args()
    
    # Initialize OCR reader
    reader = None
    if USE_EASYOCR:
        print("Initializing EasyOCR reader...")
        reader = easyocr.Reader(['en'], gpu=True)
    elif not USE_TESSERACT:
        print("Error: No OCR engine available!")
        print("Install with: pip install easyocr  OR  pip install pytesseract")
        return
    
    if args.image_dir:
        # Single directory evaluation (recursive)
        results = evaluate_directory_recursive(args.image_dir, reader, args.target_text)
        print(f"\n{'='*50}")
        print(f"Detection Rate: {results['detection_rate']:.1f}%")
        print(f"({results['detected_count']}/{results['total_images']} images)")
        print(f"{'='*50}")
        
    elif args.base_dir:
        # Multi-attack evaluation based on your directory structure
        # robustness_test/
        #   ├── attacked_blur/Center/*.png  (preprocessed images)
        #   ├── attacked_jpeg30/
        #   ├── db_blur/Center/*.png        (DreamBooth generated after preprocess)
        #   ├── db_none/
        #   └── ...
        
        if args.mode == 'db':
            # DreamBooth生成的图像（预处理后再生成）
            attack_dirs = {
                'No Attack': 'db_none',
                'JPEG Q=30': 'db_jpeg30',
                'JPEG Q=50': 'db_jpeg50',
                'Gaussian Blur': 'db_blur',
                'Gaussian Noise': 'db_noise',
                'VAE Reconstruct': 'db_vae',
                'VAE 3x': 'db_vae3x',
            }
        elif args.mode == 'attacked':
            # 直接预处理攻击后的图像
            attack_dirs = {
                'No Attack': 'attacked_none',
                'JPEG Q=30': 'attacked_jpeg30',
                'JPEG Q=50': 'attacked_jpeg50',
                'Gaussian Blur': 'attacked_blur',
                'Gaussian Noise': 'attacked_noise',
                'VAE Reconstruct': 'attacked_vae',
                'VAE 3x': 'attacked_vae3x',
            }
        else:  # all
            attack_dirs = {
                # DreamBooth generated
                'DB - No Attack': 'db_none',
                'DB - JPEG Q=30': 'db_jpeg30',
                'DB - JPEG Q=50': 'db_jpeg50',
                'DB - Blur': 'db_blur',
                'DB - Noise': 'db_noise',
                'DB - VAE': 'db_vae',
                'DB - VAE 3x': 'db_vae3x',
                # Preprocessed only
                'Attacked - None': 'attacked_none',
                'Attacked - JPEG30': 'attacked_jpeg30',
                'Attacked - JPEG50': 'attacked_jpeg50',
                'Attacked - Blur': 'attacked_blur',
                'Attacked - Noise': 'attacked_noise',
                'Attacked - VAE': 'attacked_vae',
                'Attacked - VAE 3x': 'attacked_vae3x',
            }
        
        results = evaluate_by_attack_type(args.base_dir, attack_dirs, args.target_text)
        print_summary_table(results)
        
        # Save results
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write("Visible Watermark Detection Results (OCR)\n")
            f.write(f"Target Text: {args.target_text}\n")
            f.write(f"Mode: {args.mode}\n\n")
            for attack_name, result in results.items():
                f.write(f"{attack_name}: {result['detection_rate']:.1f}% ({result['detected_count']}/{result['total_images']})\n")
        print(f"\nResults saved to {args.output}")
    
    else:
        print("Please specify --image_dir or --base_dir")
        print("\nExamples:")
        print("  # Evaluate single directory (recursive)")
        print("  python evaluate_ocr_watermark.py --image_dir ./data/robustness_test/db_none")
        print("")
        print("  # Evaluate all DreamBooth generated images")
        print("  python evaluate_ocr_watermark.py --base_dir ./data/robustness_test --mode db")
        print("")
        print("  # Evaluate all preprocessed images")
        print("  python evaluate_ocr_watermark.py --base_dir ./data/robustness_test --mode attacked")
        print("")
        print("  # Evaluate everything")
        print("  python evaluate_ocr_watermark.py --base_dir ./data/robustness_test --mode all")
        parser.print_help()


if __name__ == '__main__':
    main()
