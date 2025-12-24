from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings

config_dir = Path(__file__).parent

class Settings(BaseSettings):
    api_title: str = "3D Generation pipeline Service"

    # API settings
    host: str = "0.0.0.0"
    port: int = 10006

    # GPU settings
    qwen_gpu: int = Field(default=0, env="QWEN_GPU")
    trellis_gpu: int = Field(default=0, env="TRELLIS_GPU")
    dtype: str = Field(default="bf16", env="QWEN_DTYPE")

    # Hugging Face settings
    hf_token: Optional[str] = Field(default=None, env="HF_TOKEN")

    # Generated files settings
    save_generated_files: bool = Field(default=False, env="SAVE_GENERATED_FILES")
    send_generated_files: bool = Field(default=False, env="SEND_GENERATED_FILES")
    output_dir: Path = Field(default=Path("generated_outputs"), env="OUTPUT_DIR")

    # Trellis settings
    trellis_model_id: str = Field(default="jetx/trellis-image-large", env="TRELLIS_MODEL_ID")
    trellis_sparse_structure_steps: int = Field(default=10, env="TRELLIS_SPARSE_STRUCTURE_STEPS", description="Increased from 8 for better accuracy")
    trellis_sparse_structure_cfg_strength: float = Field(default=5.75, env="TRELLIS_SPARSE_STRUCTURE_CFG_STRENGTH")
    trellis_slat_steps: int = Field(default=25, env="TRELLIS_SLAT_STEPS", description="Increased from 20 for better quality")
    trellis_slat_cfg_strength: float = Field(default=2.4, env="TRELLIS_SLAT_CFG_STRENGTH")
    trellis_num_oversamples: int = Field(default=5, env="TRELLIS_NUM_OVERSAMPLES", description="Increased from 3 for better accuracy")
    compression: bool = Field(default=False, env="COMPRESSION")

    # Qwen Edit settings
    qwen_edit_base_model_path: str = Field(default="Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors",env="QWEN_EDIT_BASE_MODEL_PATH")
    qwen_edit_model_path: str = Field(default="Qwen/Qwen-Image-Edit-2509",env="QWEN_EDIT_MODEL_PATH")
    qwen_edit_height: int = Field(default=1024, env="QWEN_EDIT_HEIGHT")
    qwen_edit_width: int = Field(default=1024, env="QWEN_EDIT_WIDTH")
    num_inference_steps: int = Field(default=10, env="NUM_INFERENCE_STEPS", description="Increased from 8 for better quality")
    true_cfg_scale: float = Field(default=1.0, env="TRUE_CFG_SCALE")
    qwen_edit_prompt_path: Path = Field(default=config_dir.joinpath("qwen_edit_prompt.json"), env="QWEN_EDIT_PROMPT_PATH")
    qwen_edit_megapixels: float = Field(default=2.5, env="QWEN_EDIT_MEGAPIXELS", description="V2: Increased from 1.5 to 2.5 for better detail preservation")
    qwen_edit_adaptive_resolution: bool = Field(default=True, env="QWEN_EDIT_ADAPTIVE_RESOLUTION", description="V2: Use adaptive resolution based on image complexity")
    qwen_edit_detail_threshold: float = Field(default=0.8, env="QWEN_EDIT_DETAIL_THRESHOLD", description="V2: Complexity threshold for triggering higher resolution")

    # Background removal settings
    background_removal_model_id: str = Field(default="hiepnd11/rm_back2.0", env="BACKGROUND_REMOVAL_MODEL_ID")
    input_image_size: tuple[int, int] = Field(default=(1024, 1024), env="INPUT_IMAGE_SIZE") # (height, width)
    output_image_size: tuple[int, int] = Field(default=(518, 518), env="OUTPUT_IMAGE_SIZE") # (height, width)
    padding_percentage: float = Field(default=0.2, env="PADDING_PERCENTAGE")
    limit_padding: bool = Field(default=True, env="LIMIT_PADDING")
    background_mask_threshold: float = Field(default=0.8, env="BACKGROUND_MASK_THRESHOLD", description="Threshold for background mask (0.0-1.0). Higher = more strict")

    # Image editing robustness settings (V2: Enhanced)
    edit_max_retries: int = Field(default=4, env="EDIT_MAX_RETRIES", description="V2: Increased from 3 to 4")
    edit_quality_threshold: float = Field(default=0.75, env="EDIT_QUALITY_THRESHOLD", description="V2: Increased from 0.7 to 0.75")
    enable_image_preprocessing: bool = Field(default=True, env="ENABLE_IMAGE_PREPROCESSING")
    
    # V2: New quality validation settings
    use_color_validation: bool = Field(default=True, env="USE_COLOR_VALIDATION", description="V2: Validate color consistency")
    use_pattern_validation: bool = Field(default=True, env="USE_PATTERN_VALIDATION", description="V2: Validate pattern preservation")
    use_ssim_validation: bool = Field(default=True, env="USE_SSIM_VALIDATION", description="V2: Use structural similarity metric")
    
    # V2: Quality thresholds
    color_consistency_threshold: float = Field(default=0.80, env="COLOR_CONSISTENCY_THRESHOLD", description="V2: Minimum color similarity")
    pattern_preservation_threshold: float = Field(default=0.75, env="PATTERN_PRESERVATION_THRESHOLD", description="V2: Minimum pattern similarity")
    ssim_threshold: float = Field(default=0.85, env="SSIM_THRESHOLD", description="V2: Minimum structural similarity")
    edge_preservation_threshold: float = Field(default=0.70, env="EDGE_PRESERVATION_THRESHOLD", description="V2: Minimum edge similarity")
    
    # V2: Enhanced image enhancement settings
    enhancement_sharpening_factor: float = Field(default=1.1, env="ENHANCEMENT_SHARPENING_FACTOR", description="V2: Reduced from 1.2 to 1.1 (gentler)")
    enhancement_contrast_factor: float = Field(default=1.05, env="ENHANCEMENT_CONTRAST_FACTOR", description="V2: Reduced from 1.1 to 1.05 (gentler)")
    use_edge_aware_enhancement: bool = Field(default=True, env="USE_EDGE_AWARE_ENHANCEMENT", description="V2: Preserve edges during enhancement")
    
    # V2: Intelligent retry settings (optimized for <30s limit)
    intelligent_retry_enabled: bool = Field(default=True, env="INTELLIGENT_RETRY_ENABLED", description="V2: Issue-specific retry strategy")
    best_of_n_candidates: int = Field(default=2, env="BEST_OF_N_CANDIDATES", description="V2: Generate N candidates (limited to 2 for 30s constraint)")
    fast_validation_mode: bool = Field(default=False, env="FAST_VALIDATION_MODE", description="V2: Skip some validations when under time pressure")
    
    # V2: Time constraint management
    enable_time_aware_quality: bool = Field(default=True, env="ENABLE_TIME_AWARE_QUALITY", description="V2: Adjust quality based on remaining time")
    time_limit_seconds: float = Field(default=30.0, env="TIME_LIMIT_SECONDS", description="V2: Hard time limit per generation (0=disabled)")
    time_warning_threshold: float = Field(default=20.0, env="TIME_WARNING_THRESHOLD", description="V2: Start relaxing quality at this time")
    time_critical_threshold: float = Field(default=25.0, env="TIME_CRITICAL_THRESHOLD", description="V2: Emergency mode at this time")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

__all__ = ["Settings", "settings"]

