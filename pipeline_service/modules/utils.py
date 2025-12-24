from PIL import Image
from PIL import ImageStat, ImageFilter

import io
import base64
from datetime import datetime
from typing import Optional, Tuple
import os
import random
import numpy as np
import torch
import cv2

from logger_config import logger
from schemas.trellis_schemas import TrellisResult

from config import settings

def secure_randint(low: int, high: int) -> int:
    """ Return a random integer in [low, high] using os.urandom. """
    range_size = high - low + 1
    num_bytes = 4
    max_int = 2**(8 * num_bytes) - 1

    while True:
        rand_bytes = os.urandom(num_bytes)
        rand_int = int.from_bytes(rand_bytes, 'big')
        if rand_int <= max_int - (max_int % range_size):
            return low + (rand_int % range_size)

def set_random_seed(seed: int) -> None:
    """ Function for setting global seed. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def decode_image(prompt: str) -> Image.Image:
    """
    Decode the image from the base64 string.

    Args:
        prompt: The base64 string of the image.

    Returns:
        The image.
    """
    # Decode the image from the base64 string
    image_bytes = base64.b64decode(prompt)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def to_png_base64(image: Image.Image) -> str:
    """
    Convert the image to PNG format and encode to base64.

    Args:
        image: The image to convert.

    Returns:
        Base64 encoded PNG image.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    # Convert to base64 from bytes to string
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def save_file_bytes(data: bytes, folder: str, prefix: str, suffix: str) -> None:
    """
    Save binary data to the output directory.

    Args:
        data: The data to save.
        folder: The folder to save the file to.
        prefix: The prefix of the file.
        suffix: The suffix of the file.
    """
    target_dir = settings.output_dir / folder
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    path = target_dir / f"{prefix}_{timestamp}{suffix}"
    try:
        path.write_bytes(data)
        logger.debug(f"Saved file {path}")
    except Exception as exc:
        logger.error(f"Failed to save file {path}: {exc}")

def save_image(image: Image.Image, folder: str, prefix: str, timestamp: str) -> None:
    """
    Save PIL Image to the output directory.

    Args:
        image: The PIL Image to save.
        folder: The folder to save the file to.
        prefix: The prefix of the file.
        timestamp: The timestamp of the file.
    """
    target_dir = settings.output_dir / folder / timestamp
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{prefix}.png"
    try:
        image.save(path, format="PNG")
        logger.debug(f"Saved image {path}")
    except Exception as exc:
        logger.error(f"Failed to save image {path}: {exc}")

def save_files(
    trellis_result: Optional[TrellisResult], 
    image_edited: Image.Image, 
    image_without_background: Image.Image
) -> None:
    """
    Save the generated files to the output directory.

    Args:
        trellis_result: The Trellis result to save.
        image_edited: The edited image to save.
        image_without_background: The image without background to save.
    """
    # Save the Trellis result if available
    if trellis_result:
        if trellis_result.ply_file:
            save_file_bytes(trellis_result.ply_file, "ply", "mesh", suffix=".ply")

    # Save the images using PIL Image.save()
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    save_image(image_edited, "png", "image_edited", timestamp)
    save_image(image_without_background, "png", "image_without_background", timestamp)


def validate_image_quality(image: Image.Image, reference_image: Optional[Image.Image] = None) -> Tuple[bool, dict]:
    """
    Validate image quality by checking for noise, artifacts, and other issues.
    
    Args:
        image: The image to validate
        reference_image: Optional reference image to compare against
        
    Returns:
        Tuple of (is_valid, quality_metrics) where:
        - is_valid: True if image passes quality checks
        - quality_metrics: Dictionary with quality metrics
    """
    try:
        # Convert to numpy array for analysis
        img_array = np.array(image.convert("RGB"))
        
        # Calculate basic statistics
        stats = ImageStat.Stat(image)
        mean_brightness = sum(stats.mean) / len(stats.mean)
        std_dev = sum(stats.stddev) / len(stats.stddev)
        
        # Calculate variance (high variance can indicate noise)
        variance = np.var(img_array)
        
        # Calculate Laplacian variance (detects blur/noise)
        # Convert to grayscale for Laplacian
        gray = np.mean(img_array, axis=2).astype(np.float32)
        
        # Simple Laplacian-like variance calculation
        # Calculate variance of differences between adjacent pixels
        if gray.size > 1:
            h_diff = np.diff(gray, axis=0)
            w_diff = np.diff(gray, axis=1)
            laplacian_var = float(np.var(h_diff) + np.var(w_diff))
        else:
            laplacian_var = 0.0
        
        # Check for extreme values (potential artifacts)
        min_val = np.min(img_array)
        max_val = np.max(img_array)
        extreme_pixels = np.sum((img_array < 5) | (img_array > 250))
        extreme_ratio = extreme_pixels / img_array.size
        
        # Check for uniform regions (potential corruption)
        # Calculate local variance in small patches
        patch_size = 8
        h, w = gray.shape
        local_vars = []
        for i in range(0, h - patch_size, patch_size):
            for j in range(0, w - patch_size, patch_size):
                patch = gray[i:i+patch_size, j:j+patch_size]
                local_vars.append(np.var(patch))
        
        avg_local_var = np.mean(local_vars) if local_vars else 0
        
        # Quality thresholds
        quality_metrics = {
            "mean_brightness": mean_brightness,
            "std_dev": std_dev,
            "variance": float(variance),
            "laplacian_variance": float(laplacian_var),
            "extreme_pixel_ratio": extreme_ratio,
            "avg_local_variance": float(avg_local_var),
        }
        
        # Validation criteria
        is_valid = True
        issues = []
        
        # Check for excessive noise (very high variance relative to mean)
        if variance > 10000:  # Threshold for excessive noise
            is_valid = False
            issues.append("excessive_variance")
        
        # Check for too many extreme pixels (potential artifacts)
        if extreme_ratio > 0.1:  # More than 10% extreme pixels
            is_valid = False
            issues.append("excessive_extreme_pixels")
        
        # Check for too uniform (potential corruption)
        if avg_local_var < 10:  # Very low local variance
            is_valid = False
            issues.append("too_uniform")
        
        # Check for reasonable brightness range
        if mean_brightness < 10 or mean_brightness > 245:
            is_valid = False
            issues.append("extreme_brightness")
        
        quality_metrics["is_valid"] = is_valid
        quality_metrics["issues"] = issues
        
        return is_valid, quality_metrics
        
    except Exception as e:
        logger.warning(f"Error validating image quality: {e}")
        # On error, assume valid but log the issue
        return True, {"error": str(e), "is_valid": True}


def preprocess_input_image(image: Image.Image) -> Image.Image:
    """
    Preprocess input image to improve editing quality.
    
    Args:
        image: Input image to preprocess
        
    Returns:
        Preprocessed image
    """
    try:
        # Ensure RGB format
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Get image statistics
        stats = ImageStat.Stat(image)
        mean_brightness = sum(stats.mean) / len(stats.mean)
        
        # Normalize brightness if too dark or too bright
        if mean_brightness < 30:
            # Too dark - apply slight brightening
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.2)
            logger.debug("Applied brightness enhancement to dark image")
        elif mean_brightness > 225:
            # Too bright - apply slight darkening
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(0.9)
            logger.debug("Applied brightness reduction to bright image")
        
        return image
        
    except Exception as e:
        logger.warning(f"Error preprocessing image: {e}, returning original")
        return image.convert("RGB") if image.mode != "RGB" else image


# ==================== V2: New Validation Functions ====================

def calculate_ssim(image1: Image.Image, image2: Image.Image) -> float:
    """
    Calculate Structural Similarity Index (SSIM) between two images.
    
    Args:
        image1: First image
        image2: Second image
        
    Returns:
        SSIM score (0.0 to 1.0, higher is better)
    """
    try:
        # Convert to numpy arrays
        img1 = np.array(image1.convert("RGB"), dtype=np.float32)
        img2 = np.array(image2.convert("RGB"), dtype=np.float32)
        
        # Resize if dimensions don't match
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Calculate SSIM for each channel
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        
        # Calculate means
        mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # Calculate variances and covariance
        sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
        
        # Calculate SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return float(np.mean(ssim_map))
        
    except Exception as e:
        logger.warning(f"Error calculating SSIM: {e}")
        return 1.0  # Assume valid on error


def validate_color_consistency(original: Image.Image, edited: Image.Image) -> Tuple[bool, dict]:
    """
    Validate color consistency between original and edited images.
    Detects color shifts in jewels, patterns, etc.
    
    Args:
        original: Original image
        edited: Edited image
        
    Returns:
        Tuple of (is_valid, metrics) where metrics contains color analysis
    """
    try:
        # Convert to RGB
        orig_rgb = np.array(original.convert("RGB"), dtype=np.float32)
        edit_rgb = np.array(edited.convert("RGB"), dtype=np.float32)
        
        # Resize edited to match original if needed
        if orig_rgb.shape != edit_rgb.shape:
            edit_rgb = cv2.resize(edit_rgb, (orig_rgb.shape[1], orig_rgb.shape[0]))
        
        # 1. Color histogram comparison
        hist_similarity = 0.0
        for channel in range(3):
            hist_orig = cv2.calcHist([orig_rgb], [channel], None, [256], [0, 256])
            hist_edit = cv2.calcHist([edit_rgb], [channel], None, [256], [0, 256])
            
            # Normalize histograms
            hist_orig = hist_orig / hist_orig.sum()
            hist_edit = hist_edit / hist_edit.sum()
            
            # Calculate correlation
            correlation = cv2.compareHist(hist_orig, hist_edit, cv2.HISTCMP_CORREL)
            hist_similarity += correlation
        
        hist_similarity /= 3.0  # Average across channels
        
        # 2. Dominant color preservation
        orig_dominant = orig_rgb.reshape(-1, 3).mean(axis=0)
        edit_dominant = edit_rgb.reshape(-1, 3).mean(axis=0)
        color_shift = np.linalg.norm(orig_dominant - edit_dominant) / 255.0
        
        # 3. Color variance preservation
        orig_variance = np.var(orig_rgb, axis=(0, 1)).mean()
        edit_variance = np.var(edit_rgb, axis=(0, 1)).mean()
        variance_ratio = min(orig_variance, edit_variance) / max(orig_variance, edit_variance) if max(orig_variance, edit_variance) > 0 else 1.0
        
        # Combined color consistency score
        color_consistency_score = (hist_similarity * 0.5 + (1.0 - color_shift) * 0.3 + variance_ratio * 0.2)
        
        metrics = {
            "color_consistency_score": float(color_consistency_score),
            "histogram_similarity": float(hist_similarity),
            "color_shift": float(color_shift),
            "variance_ratio": float(variance_ratio),
        }
        
        # Validate against threshold
        threshold = settings.color_consistency_threshold if hasattr(settings, 'color_consistency_threshold') else 0.80
        is_valid = color_consistency_score >= threshold
        
        if not is_valid:
            metrics["issues"] = [f"color_consistency_low_{color_consistency_score:.2f}"]
        
        return is_valid, metrics
        
    except Exception as e:
        logger.warning(f"Error validating color consistency: {e}")
        return True, {"error": str(e), "color_consistency_score": 1.0}


def validate_pattern_preservation(original: Image.Image, edited: Image.Image) -> Tuple[bool, dict]:
    """
    Validate that fine patterns and textures are preserved.
    Important for detecting loss of details like flowers, fabric patterns, etc.
    
    Args:
        original: Original image
        edited: Edited image
        
    Returns:
        Tuple of (is_valid, metrics) containing pattern analysis
    """
    try:
        # Convert to grayscale for pattern analysis
        orig_gray = cv2.cvtColor(np.array(original.convert("RGB")), cv2.COLOR_RGB2GRAY)
        edit_gray = cv2.cvtColor(np.array(edited.convert("RGB")), cv2.COLOR_RGB2GRAY)
        
        # Resize if needed
        if orig_gray.shape != edit_gray.shape:
            edit_gray = cv2.resize(edit_gray, (orig_gray.shape[1], orig_gray.shape[0]))
        
        # 1. Edge density comparison (Canny edge detection)
        orig_edges = cv2.Canny(orig_gray, 50, 150)
        edit_edges = cv2.Canny(edit_gray, 50, 150)
        
        orig_edge_density = np.sum(orig_edges > 0) / orig_edges.size
        edit_edge_density = np.sum(edit_edges > 0) / edit_edges.size
        
        edge_preservation = min(orig_edge_density, edit_edge_density) / max(orig_edge_density, edit_edge_density) if max(orig_edge_density, edit_edge_density) > 0 else 1.0
        
        # 2. High-frequency detail preservation (using Laplacian)
        orig_laplacian = cv2.Laplacian(orig_gray, cv2.CV_64F)
        edit_laplacian = cv2.Laplacian(edit_gray, cv2.CV_64F)
        
        orig_detail_variance = np.var(orig_laplacian)
        edit_detail_variance = np.var(edit_laplacian)
        
        detail_preservation = min(orig_detail_variance, edit_detail_variance) / max(orig_detail_variance, edit_detail_variance) if max(orig_detail_variance, edit_detail_variance) > 0 else 1.0
        
        # 3. Texture similarity using local standard deviation
        orig_texture = cv2.blur(orig_gray, (5, 5))
        edit_texture = cv2.blur(edit_gray, (5, 5))
        
        texture_diff = np.abs(orig_texture.astype(float) - edit_texture.astype(float)).mean()
        texture_similarity = 1.0 - min(texture_diff / 255.0, 1.0)
        
        # Combined pattern preservation score
        pattern_score = (edge_preservation * 0.4 + detail_preservation * 0.3 + texture_similarity * 0.3)
        
        metrics = {
            "pattern_preservation_score": float(pattern_score),
            "edge_preservation": float(edge_preservation),
            "detail_preservation": float(detail_preservation),
            "texture_similarity": float(texture_similarity),
            "orig_edge_density": float(orig_edge_density),
            "edit_edge_density": float(edit_edge_density),
        }
        
        # Validate against threshold
        threshold = settings.pattern_preservation_threshold if hasattr(settings, 'pattern_preservation_threshold') else 0.75
        is_valid = pattern_score >= threshold
        
        if not is_valid:
            metrics["issues"] = [f"pattern_preservation_low_{pattern_score:.2f}"]
        
        return is_valid, metrics
        
    except Exception as e:
        logger.warning(f"Error validating pattern preservation: {e}")
        return True, {"error": str(e), "pattern_preservation_score": 1.0}


def calculate_image_complexity(image: Image.Image) -> float:
    """
    Calculate image complexity score to determine if adaptive resolution is needed.
    High complexity images (jewels, patterns) benefit from higher resolution.
    
    Args:
        image: Input image
        
    Returns:
        Complexity score (0.0 to 1.0, higher = more complex)
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        
        # 1. Edge density (complex images have more edges)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # 2. Color variance (complex images have more color variation)
        rgb = np.array(image.convert("RGB"))
        color_variance = np.var(rgb, axis=(0, 1)).mean() / 10000.0  # Normalize
        
        # 3. Texture complexity (using Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_complexity = min(np.var(laplacian) / 1000.0, 1.0)  # Normalize
        
        # 4. Frequency domain analysis (FFT)
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        high_freq_energy = np.sum(magnitude[magnitude > np.percentile(magnitude, 90)]) / magnitude.size
        high_freq_score = min(high_freq_energy / 1000.0, 1.0)
        
        # Combined complexity score
        complexity = (edge_density * 0.3 + color_variance * 0.2 + texture_complexity * 0.3 + high_freq_score * 0.2)
        
        return float(min(complexity, 1.0))
        
    except Exception as e:
        logger.warning(f"Error calculating complexity: {e}")
        return 0.5  # Default to medium complexity


def calculate_adaptive_megapixels(image: Image.Image, base_megapixels: float = 2.5) -> float:
    """
    Calculate adaptive megapixels based on image complexity.
    
    Args:
        image: Input image
        base_megapixels: Base megapixel value
        
    Returns:
        Adjusted megapixels value
    """
    try:
        complexity = calculate_image_complexity(image)
        
        # High complexity: increase resolution
        # Low complexity: can use lower resolution
        if complexity > 0.8:
            # Very complex (jewels, fine patterns): +20%
            return base_megapixels * 1.2
        elif complexity > 0.6:
            # Medium-high complexity: +10%
            return base_megapixels * 1.1
        elif complexity < 0.3:
            # Low complexity: -10%
            return base_megapixels * 0.9
        else:
            # Medium complexity: use base
            return base_megapixels
            
    except Exception as e:
        logger.warning(f"Error calculating adaptive megapixels: {e}")
        return base_megapixels


def validate_image_quality_v2(image: Image.Image, reference_image: Optional[Image.Image] = None) -> Tuple[bool, dict]:
    """
    V2: Enhanced image quality validation with multi-metric scoring.
    
    Args:
        image: The image to validate
        reference_image: Optional reference image to compare against
        
    Returns:
        Tuple of (is_valid, quality_metrics)
    """
    try:
        # Get V1 metrics
        is_valid_v1, metrics_v1 = validate_image_quality(image, reference_image)
        
        # If reference image provided, calculate additional V2 metrics
        if reference_image is not None:
            # SSIM
            if settings.use_ssim_validation if hasattr(settings, 'use_ssim_validation') else True:
                ssim_score = calculate_ssim(reference_image, image)
                metrics_v1["ssim_score"] = ssim_score
                ssim_threshold = settings.ssim_threshold if hasattr(settings, 'ssim_threshold') else 0.85
                if ssim_score < ssim_threshold:
                    is_valid_v1 = False
                    metrics_v1.setdefault("issues", []).append(f"low_ssim_{ssim_score:.2f}")
            
            # Color consistency
            if settings.use_color_validation if hasattr(settings, 'use_color_validation') else True:
                color_valid, color_metrics = validate_color_consistency(reference_image, image)
                metrics_v1.update(color_metrics)
                if not color_valid:
                    is_valid_v1 = False
                    metrics_v1.setdefault("issues", []).extend(color_metrics.get("issues", []))
            
            # Pattern preservation
            if settings.use_pattern_validation if hasattr(settings, 'use_pattern_validation') else True:
                pattern_valid, pattern_metrics = validate_pattern_preservation(reference_image, image)
                metrics_v1.update(pattern_metrics)
                if not pattern_valid:
                    is_valid_v1 = False
                    metrics_v1.setdefault("issues", []).extend(pattern_metrics.get("issues", []))
            
            # Calculate weighted quality score
            weights = {
                "ssim": 0.25,
                "color_consistency": 0.25,
                "pattern_preservation": 0.25,
                "variance": 0.15,
                "edge_preservation": 0.10
            }
            
            quality_components = []
            if "ssim_score" in metrics_v1:
                quality_components.append(metrics_v1["ssim_score"] * weights["ssim"])
            if "color_consistency_score" in metrics_v1:
                quality_components.append(metrics_v1["color_consistency_score"] * weights["color_consistency"])
            if "pattern_preservation_score" in metrics_v1:
                quality_components.append(metrics_v1["pattern_preservation_score"] * weights["pattern_preservation"])
            
            if quality_components:
                metrics_v1["weighted_quality_score"] = sum(quality_components)
        
        return is_valid_v1, metrics_v1
        
    except Exception as e:
        logger.warning(f"Error in V2 quality validation: {e}")
        # Fallback to V1
        return validate_image_quality(image, reference_image)

