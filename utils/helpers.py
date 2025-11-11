"""
Genesis Helper Utilities
Helper utility functions
"""

import os
import random
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from datetime import datetime


def generate_seed(seed: Optional[int] = None) -> int:
    """
    Generate or validate seed
    
    Args:
        seed: Input seed (-1 or None for random)
        
    Returns:
        Valid seed value
    """
    if seed is None or seed == -1:
        return random.randint(0, 2**32 - 1)
    return seed


def hash_string(text: str) -> str:
    """
    Generate hash from string
    
    Args:
        text: Input text
        
    Returns:
        MD5 hash string
    """
    return hashlib.md5(text.encode()).hexdigest()


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_timestamp() -> str:
    """
    Get current timestamp string
    
    Returns:
        Timestamp string (YYYYMMDD_HHMMSS)
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes to human readable string
    
    Args:
        bytes_value: Byte count
        
    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def format_time(seconds: float) -> str:
    """
    Format seconds to readable time string
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing invalid characters
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename


def truncate_string(text: str, max_length: int = 50, suffix: str = "...") -> str:
    """
    Truncate string to max length
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to append
        
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def parse_resolution(resolution: str) -> tuple[int, int]:
    """
    Parse resolution string to (width, height)
    
    Args:
        resolution: Resolution string (e.g., "512x512", "1024x768")
        
    Returns:
        (width, height) tuple
        
    Raises:
        ValueError: If resolution format is invalid
    """
    try:
        parts = resolution.lower().split('x')
        if len(parts) != 2:
            raise ValueError("Invalid resolution format")
        width = int(parts[0])
        height = int(parts[1])
        return (width, height)
    except (ValueError, IndexError):
        raise ValueError(f"Invalid resolution: {resolution}. Use format like '512x512'")


def get_file_size(file_path: Union[str, Path]) -> int:
    """
    Get file size in bytes
    
    Args:
        file_path: File path
        
    Returns:
        File size in bytes
    """
    return Path(file_path).stat().st_size


def list_files(
    directory: Union[str, Path],
    extensions: Optional[List[str]] = None,
    recursive: bool = False
) -> List[Path]:
    """
    List files in directory
    
    Args:
        directory: Directory path
        extensions: File extensions to filter (e.g., ['.safetensors', '.ckpt'])
        recursive: Search recursively
        
    Returns:
        List of file paths
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    
    files = []
    pattern = "**/*" if recursive else "*"
    
    for file_path in directory.glob(pattern):
        if not file_path.is_file():
            continue
        
        if extensions is None or file_path.suffix.lower() in extensions:
            files.append(file_path)
    
    return sorted(files)


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries
    
    Args:
        *dicts: Dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result


def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Clamp value between min and max
    
    Args:
        value: Input value
        min_value: Minimum value
        max_value: Maximum value
        
    Returns:
        Clamped value
    """
    return max(min_value, min(max_value, value))


def round_to_multiple(value: int, multiple: int) -> int:
    """
    Round value to nearest multiple
    
    Args:
        value: Input value
        multiple: Multiple to round to
        
    Returns:
        Rounded value
    """
    return ((value + multiple - 1) // multiple) * multiple


class ProgressTracker:
    """Simple progress tracker"""
    
    def __init__(self, total: int, description: str = ""):
        self.total = total
        self.current = 0
        self.description = description
        
    def update(self, n: int = 1):
        """Update progress"""
        self.current += n
        self.display()
        
    def display(self):
        """Display progress"""
        percentage = (self.current / self.total) * 100 if self.total > 0 else 0
        bar_length = 40
        filled = int(bar_length * self.current / self.total) if self.total > 0 else 0
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f"\r{self.description} [{bar}] {percentage:.1f}% ({self.current}/{self.total})", end='', flush=True)
        
        if self.current >= self.total:
            print()  # New line when complete
    
    def reset(self):
        """Reset progress"""
        self.current = 0


# Preset resolutions
PRESET_RESOLUTIONS = {
    'sd15': [(512, 512), (768, 512), (512, 768), (640, 640)],
    'sdxl': [(1024, 1024), (1216, 832), (832, 1216), (1344, 768), (768, 1344)],
    'sd3': [(1024, 1024), (1152, 896), (896, 1152), (1216, 832), (832, 1216)]
}


def get_preset_resolutions(model_type: str = 'sd15') -> List[tuple[int, int]]:
    """
    Get preset resolutions for model type
    
    Args:
        model_type: Model type ('sd15', 'sdxl', 'sd3')
        
    Returns:
        List of (width, height) tuples
    """
    return PRESET_RESOLUTIONS.get(model_type, PRESET_RESOLUTIONS['sd15'])
