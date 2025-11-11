"""
Genesis Configuration
Configuration management system
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
from pathlib import Path


@dataclass
class GenesisConfig:
    """Genesis Engine Configuration"""
    
    # Device configuration
    device: Literal['cuda', 'cpu', 'mps'] = 'cuda'
    device_id: int = 0
    
    # Memory management
    vram_mode: Literal['auto', 'low', 'normal', 'high'] = 'auto'
    max_vram_gb: Optional[float] = None
    
    # Model paths
    models_dir: Path = field(default_factory=lambda: Path('models'))
    checkpoints_dir: Path = field(default_factory=lambda: Path('models/checkpoints'))
    vae_dir: Path = field(default_factory=lambda: Path('models/vae'))
    lora_dir: Path = field(default_factory=lambda: Path('models/loras'))
    embeddings_dir: Path = field(default_factory=lambda: Path('models/embeddings'))
    
    # Output configuration
    output_dir: Path = field(default_factory=lambda: Path('output'))
    temp_dir: Path = field(default_factory=lambda: Path('temp'))
    
    # Performance configuration
    num_threads: int = 4
    pin_memory: bool = True
    allow_tf32: bool = True
    
    # Cache configuration
    enable_cache: bool = True
    cache_size_mb: int = 2048
    
    # Logging configuration
    log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR'] = 'INFO'
    log_to_file: bool = False
    log_file: Path = field(default_factory=lambda: Path('genesis.log'))
    
    def __post_init__(self):
        """Post-initialization processing"""
        # Ensure paths are Path objects
        for field_name in ['models_dir', 'checkpoints_dir', 'vae_dir', 'lora_dir', 
                          'embeddings_dir', 'output_dir', 'temp_dir', 'log_file']:
            value = getattr(self, field_name)
            if not isinstance(value, Path):
                setattr(self, field_name, Path(value))
    
    def create_directories(self):
        """Create necessary directories"""
        dirs = [
            self.models_dir,
            self.checkpoints_dir,
            self.vae_dir,
            self.lora_dir,
            self.embeddings_dir,
            self.output_dir,
            self.temp_dir,
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'GenesisConfig':
        """Create configuration from dictionary"""
        return cls(**config_dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'device': self.device,
            'device_id': self.device_id,
            'vram_mode': self.vram_mode,
            'max_vram_gb': self.max_vram_gb,
            'models_dir': str(self.models_dir),
            'checkpoints_dir': str(self.checkpoints_dir),
            'vae_dir': str(self.vae_dir),
            'lora_dir': str(self.lora_dir),
            'embeddings_dir': str(self.embeddings_dir),
            'output_dir': str(self.output_dir),
            'temp_dir': str(self.temp_dir),
            'num_threads': self.num_threads,
            'pin_memory': self.pin_memory,
            'allow_tf32': self.allow_tf32,
            'enable_cache': self.enable_cache,
            'cache_size_mb': self.cache_size_mb,
            'log_level': self.log_level,
            'log_to_file': self.log_to_file,
            'log_file': str(self.log_file),
        }
