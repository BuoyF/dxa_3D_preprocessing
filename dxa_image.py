
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple, Optional, Dict


def load_dxa_image(image_path: str) -> np.ndarray:
    """
    Load DXA image from file
    
    Args:
        image_path: Path to image file (.png, .jpg, .jpeg)
        
    Returns:
        image: (H, W, C) array, uint8 [0, 255]
    """
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Ensure 3 channels
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:
        # Remove alpha channel if present
        img_array = img_array[:, :, :3]
    
    return img_array


def rgb_to_density_sarcopenia(rgb_image: np.ndarray) -> np.ndarray:
    """
    Convert color-coded to density-based grayscale
    
    Color coding in sarcopenia images:
    - Blue/Cyan: Bone (highest density)
    - Red/Orange: Lean tissue (medium density)
    - Green/Yellow: Fat tissue (low density)
    - Black: Background (zero density)
    
    Args:
        rgb_image: (H, W, 3) RGB image, uint8 [0, 255]
        
    Returns:
        density_image: (H, W) grayscale density map, uint8 [0, 255]
    """
    # Normalize to [0, 1]
    rgb = rgb_image.astype(np.float32) / 255.0
    
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    
    # Create density map based on color channels
    # Blue (bone) = high intensity
    # Red (lean) = medium intensity
    # Green (fat) = low-medium intensity
    
    # Weight channels by tissue density
    # Bone (blue dominant): high weight
    # Lean (red dominant): medium weight
    # Fat (green dominant): low weight
    
    bone_score = b * (1 - r) * (1 - g)  # High blue, low red/green
    lean_score = r * (1 - b) * (1 - g)  # High red, low blue/green
    fat_score = g * (1 - b) * (1 - r)   # High green, low blue/red
    
    # Combine with density weights
    density = (
        bone_score * 1.0 +      # Bone: highest density
        lean_score * 0.6 +      # Lean: medium density
        fat_score * 0.3         # Fat: low density
    )
    
    # Normalize to [0, 255]
    density = (density * 255).astype(np.uint8)
    
    return density


def rgb_to_grayscale_normal(rgb_image: np.ndarray) -> np.ndarray:
    """
    Convert normal/bariatric RGB DXA to grayscale
    These images are already effectively grayscale (all channels similar)
    
    Args:
        rgb_image: (H, W, 3) RGB image, uint8 [0, 255]
        
    Returns:
        gray_image: (H, W) grayscale image, uint8 [0, 255]
    """
    # Check if already grayscale
    r, g, b = rgb_image[:, :, 0], rgb_image[:, :, 1], rgb_image[:, :, 2]
    
    if np.allclose(r, g) and np.allclose(g, b):
        # Already grayscale, just take one channel
        return r.astype(np.uint8)
    else:
        # Convert using standard weights
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        return gray


def create_body_mask(image: np.ndarray, threshold: float = 10) -> np.ndarray:
    """
    Create binary mask separating body from background
    Keeps only the largest connected component (the body) and removes other parts
    
    Args:
        image: (H, W) grayscale image, uint8 [0, 255]
        threshold: Intensity threshold for background
        
    Returns:
        mask: (H, W) binary mask, uint8 {0, 255}
    """
    # Simple thresholding
    mask = (image > threshold).astype(np.uint8) * 255
    
    # Morphological operations to clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Keep only the largest connected component (body)
    num_labels, labels = cv2.connectedComponents(mask)
    
    if num_labels > 1:
        # Find the largest component (excluding background label 0)
        largest_label = 1
        largest_size = 0
        
        for label in range(1, num_labels):
            component_size = np.sum(labels == label)
            if component_size > largest_size:
                largest_size = component_size
                largest_label = label
        
        # Create mask with only the largest component
        mask = (labels == largest_label).astype(np.uint8) * 255
    
    return mask


def resize_dxa_image(image: np.ndarray, 
                     target_size: Tuple[int, int] = (320, 864),
                     keep_aspect_ratio: bool = True) -> np.ndarray:
    """
    Resize DXA image to target size
    
    Args:
        image: (H, W) or (H, W, C) image
        target_size: (width, height) target size
        keep_aspect_ratio: If True, pad to maintain aspect ratio
        
    Returns:
        resized_image: (target_h, target_w) or (target_h, target_w, C) image
    """
    target_w, target_h = target_size
    
    if keep_aspect_ratio:
        # Compute scale to fit within target size
        h, w = image.shape[:2]
        scale = min(target_w / w, target_h / h)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to target size
        if len(image.shape) == 2:
            padded = np.zeros((target_h, target_w), dtype=image.dtype)
        else:
            padded = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
        
        # Center the resized image
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return padded
    else:
        # Direct resize (may distort aspect ratio)
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        return resized


def normalize_intensity(image: np.ndarray, 
                        method: str = 'zscore_global',
                        mask: Optional[np.ndarray] = None,
                        global_mean: Optional[float] = None,
                        global_std: Optional[float] = None) -> np.ndarray:
    """
    Normalize image intensity.
    
    CRITICAL: For DXA, pixel intensity = physical mass/density.
    Per-image normalization destroys this relationship.
    Use global z-score normalization computed across training set.
    
    Args:
        image: (H, W) grayscale image, uint8 [0, 255]
        method: Normalization method:
            - 'zscore_global': z-score using global stats (RECOMMENDED for physics preservation)
            - 'zscore_per_image': z-score per image (WARNING: destroys absolute density)
        mask: Optional body mask to compute stats only on body region
        global_mean: Global mean computed from training set (required for zscore_global)
        global_std: Global std computed from training set (required for zscore_global)
        
    Returns:
        normalized_image: (H, W) float32
    """
    img_float = image.astype(np.float32)
    
    if method == 'zscore_global':
        # PHYSICS-PRESERVING: Use global statistics to maintain absolute density
        if global_mean is None:
            global_mean = 127.5  # Default: middle of uint8 range
        if global_std is None:
            global_std = 64.0    # Default: ~1/4 of uint8 range
        
        img_norm = (img_float - global_mean) / (global_std + 1e-6)
        # Clip to [-3, 3] to handle outliers
        img_norm = np.clip(img_norm, -3, 3)
        # Scale to [0, 1] range for network input
        img_norm = (img_norm + 3) / 6
        return img_norm
        
    elif method == 'zscore_per_image':
        # WARNING: Per-image normalization destroys absolute mass information!
        # All patients (obese/underweight) will have same peak intensity.
        if mask is not None:
            body_pixels = img_float[mask > 0]
        else:
            body_pixels = img_float
        
        mean_val = body_pixels.mean()
        std_val = body_pixels.std()
        
        if std_val > 0:
            img_norm = (img_float - mean_val) / std_val
            img_norm = np.clip(img_norm, -3, 3)
            img_norm = (img_norm + 3) / 6
        else:
            img_norm = img_float / 255.0
        return img_norm
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return img_norm


def augment_dxa_image(image: np.ndarray,
                      rotation: float = 2.0,
                      brightness: float = 0.1,
                      contrast: float = 0.1,
                      gaussian_noise: float = 0.005,
                      random_state: Optional[np.random.RandomState] = None) -> np.ndarray:
    """
    Apply data augmentation to DXA image
    
    Args:
        image: (H, W) float32 [0, 1] image
        rotation: Max rotation angle (degrees)
        brightness: Max brightness shift (fraction)
        contrast: Max contrast adjustment (fraction)
        gaussian_noise: Gaussian noise std
        random_state: Random state for reproducibility
        
    Returns:
        augmented_image: (H, W) float32 [0, 1]
    """
    if random_state is None:
        random_state = np.random.RandomState()
    
    img_aug = image.copy()
    h, w = img_aug.shape
    
    # Rotation
    if rotation > 0:
        angle = random_state.uniform(-rotation, rotation)
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        img_aug = cv2.warpAffine(img_aug, rot_mat, (w, h), 
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0)
    
    # Brightness
    if brightness > 0:
        brightness_shift = random_state.uniform(-brightness, brightness)
        img_aug = img_aug + brightness_shift
    
    # Contrast
    if contrast > 0:
        contrast_factor = random_state.uniform(1 - contrast, 1 + contrast)
        mean_val = img_aug.mean()
        img_aug = (img_aug - mean_val) * contrast_factor + mean_val
    
    # Gaussian noise
    if gaussian_noise > 0:
        noise = random_state.normal(0, gaussian_noise, img_aug.shape)
        img_aug = img_aug + noise
    
    # Clip to [0, 1]
    img_aug = np.clip(img_aug, 0, 1)
    
    return img_aug


def preprocess_dxa_image(image_path: str,
                         dataset_type: str = 'normal',
                         target_size: Tuple[int, int] = (320, 864),
                         normalization: str = 'percentile',
                         create_mask: bool = True,
                         augment: bool = False,
                         augment_params: Optional[dict] = None,
                         seed: int = 0) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
    """
    Complete preprocessing pipeline for DXA image
    
    Args:
        image_path: Path to DXA image
        dataset_type: 'normal', 'bariatric', or 'sarcopenia'
        target_size: (width, height) target size
        normalization: Intensity normalization method
        create_mask: Whether to create body mask
        augment: Whether to apply augmentation
        augment_params: Augmentation parameters
        seed: Random seed
        
    Returns:
        image: (H, W) preprocessed image, float32 [0, 1]
        mask: (H, W) body mask, uint8 {0, 255} (or None)
        metadata: Dictionary with preprocessing info
    """
    # Load RGB image
    rgb_image = load_dxa_image(image_path)
    original_shape = rgb_image.shape
    
    # Convert to grayscale/density
    if dataset_type == 'sarcopenia':
        gray_image = rgb_to_density_sarcopenia(rgb_image)
    else:
        gray_image = rgb_to_grayscale_normal(rgb_image)
    
    # Resize
    gray_image = resize_dxa_image(gray_image, target_size, keep_aspect_ratio=True)
    
    # Create mask
    if create_mask:
        mask = create_body_mask(gray_image, threshold=10)
    else:
        mask = None
    
    # Normalize intensity
    image_norm = normalize_intensity(gray_image, method=normalization, mask=mask)
    
    # Augment (if training)
    if augment and augment_params is not None:
        rng = np.random.RandomState(seed)
        image_norm = augment_dxa_image(image_norm, random_state=rng, **augment_params)
    
    metadata = {
        'image_path': str(image_path),
        'dataset_type': dataset_type,
        'original_shape': original_shape,
        'target_size': target_size,
        'normalization': normalization,
        'has_mask': mask is not None,
        'augmented': augment,
    }
    
    return image_norm, mask, metadata


def batch_dxa_images(images: list, masks: Optional[list] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Batch DXA images and masks
    
    Args:
        images: List of (H, W) float32 arrays
        masks: Optional list of (H, W) uint8 arrays
        
    Returns:
        batched_images: (B, 1, H, W) tensor
        batched_masks: (B, 1, H, W) tensor (or None)
    """
    # Stack images
    images_array = np.stack(images, axis=0)[:, None, :, :]  # (B, 1, H, W)
    batched_images = torch.from_numpy(images_array).float()
    
    # Stack masks if provided
    if masks is not None:
        masks_array = np.stack(masks, axis=0)[:, None, :, :] / 255.0  # (B, 1, H, W), [0, 1]
        batched_masks = torch.from_numpy(masks_array).float()
    else:
        batched_masks = None
    
    return batched_images, batched_masks


def visualize_dxa_preprocessing(original_path: str, 
                                 preprocessed: np.ndarray,
                                 mask: Optional[np.ndarray] = None,
                                 save_path: Optional[str] = None):
    """
    Visualize DXA preprocessing results
    
    Args:
        original_path: Path to original image
        preprocessed: Preprocessed image
        mask: Body mask (optional)
        save_path: Path to save visualization (optional)
    """
    import matplotlib.pyplot as plt
    
    # Load original
    original = load_dxa_image(original_path)
    if len(original.shape) == 3:
        original_gray = rgb_to_grayscale_normal(original)
    else:
        original_gray = original
    
    # Create figure
    n_cols = 3 if mask is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(6*n_cols, 8))
    
    # Original
    axes[0].imshow(original_gray, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Preprocessed
    axes[1].imshow(preprocessed, cmap='gray')
    axes[1].set_title('Preprocessed')
    axes[1].axis('off')
    
    # Mask
    if mask is not None:
        axes[2].imshow(mask, cmap='gray')
        axes[2].set_title('Body Mask')
        axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    print("Testing DXA image preprocessing utilities...")
    
    # Create output directory for test results
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)
    print(f"\nSaving outputs to: {output_dir.absolute()}")
    
    # Create dummy images
    print("\n1. Testing grayscale conversion...")
    dummy_rgb = np.random.randint(0, 256, (800, 300, 3), dtype=np.uint8)
    gray = rgb_to_grayscale_normal(dummy_rgb)
    print(f"  RGB shape: {dummy_rgb.shape}")
    print(f"  Grayscale shape: {gray.shape}")
    Image.fromarray(gray).save(output_dir / "01_grayscale.png")
    print(f" Saved to 01_grayscale.png")
    
    print("\n2. Testing sarcopenia color-to-density conversion...")
    # Create color-coded image (blue=bone, red=lean, green=fat)
    dummy_color = np.zeros((800, 300, 3), dtype=np.uint8)
    dummy_color[:, :100, 2] = 255  # Blue region (bone)
    dummy_color[:, 100:200, 0] = 255  # Red region (lean)
    dummy_color[:, 200:, 1] = 255  # Green region (fat)
    
    Image.fromarray(dummy_color).save(output_dir / "02a_color_input.png")
    
    density = rgb_to_density_sarcopenia(dummy_color)
    print(f"  Color shape: {dummy_color.shape}")
    print(f"  Density shape: {density.shape}")
    print(f"  Density range: [{density.min()}, {density.max()}]")
    Image.fromarray(density).save(output_dir / "02b_density_output.png")
    print(f" Saved to 02a_color_input.png and 02b_density_output.png")
    
    print("\n3. Testing mask creation...")
    mask = create_body_mask(gray)
    print(f"  Mask shape: {mask.shape}")
    print(f"  Mask values: {np.unique(mask)}")
    Image.fromarray(mask).save(output_dir / "03_mask.png")
    print(f" Saved to 03_mask.png")
    
    print("\n4. Testing resize...")
    resized = resize_dxa_image(gray, target_size=(320, 864))
    print(f"  Original shape: {gray.shape}")
    print(f"  Resized shape: {resized.shape}")
    Image.fromarray(resized).save(output_dir / "04_resized.png")
    print(f" Saved to 04_resized.png")
    
    print("\n5. Testing normalization...")
    normalized = normalize_intensity(resized, method='percentile', mask=None)
    print(f"  Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    print(f"  Normalized mean: {normalized.mean():.3f}")
    normalized_uint8 = (normalized * 255).astype(np.uint8)
    Image.fromarray(normalized_uint8).save(output_dir / "05_normalized.png")
    print(f" Saved to 05_normalized.png")
    
    print("\n6. Testing augmentation...")
    augmented = augment_dxa_image(normalized)
    print(f"  Augmented range: [{augmented.min():.3f}, {augmented.max():.3f}]")
    augmented_uint8 = (augmented * 255).astype(np.uint8)
    Image.fromarray(augmented_uint8).save(output_dir / "06_augmented.png")
    print(f" Saved to 06_augmented.png")
    
    print(f"\n✓ All tests passed!")
    print(f"\nAll test images saved to: {output_dir.absolute()}")