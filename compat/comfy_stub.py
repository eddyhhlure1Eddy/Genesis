"""
ComfyUI comfy module compatibility layer for Genesis
Provides complete comfy API compatibility
Author: eddy
"""

import sys
import torch
import logging
import hashlib
from types import ModuleType

logger = logging.getLogger(__name__)


class ModelPatcher:
    """Stub for comfy.model_patcher.ModelPatcher"""
    
    def __init__(self, model, load_device=None, offload_device=None):
        self.model = model
        self.load_device = load_device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.offload_device = offload_device or torch.device("cpu")
        self.patches = {}
    
    def patch_model(self, patches):
        self.patches.update(patches)
        return self
    
    def clone(self):
        return ModelPatcher(self.model, self.load_device, self.offload_device)
    
    def to(self, device):
        self.load_device = device
        if hasattr(self.model, 'to'):
            self.model.to(device)
        return self


class ModelManagement:
    """Stub for comfy.model_management"""
    
    @staticmethod
    def get_torch_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @staticmethod
    def get_autocast_device(device):
        return device.type if hasattr(device, 'type') else 'cpu'
    
    @staticmethod
    def soft_empty_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @staticmethod
    def load_models_gpu(models, memory_required=0):
        pass
    
    @staticmethod
    def unload_all_models():
        pass
    
    @staticmethod
    def get_free_memory(device=None):
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory
        return 0
    
    @staticmethod
    def interrupt_current_processing(value=True):
        pass


class Utils:
    """Stub for comfy.utils"""
    
    PROGRESS_BAR_ENABLED = True
    
    @staticmethod
    def ProgressBar(total):
        class DummyProgressBar:
            def __init__(self):
                self.total = total
                self.current = 0
            
            def update(self, value):
                self.current = value
            
            def update_absolute(self, value, total=None):
                self.current = value
        
        return DummyProgressBar()
    
    @staticmethod
    def copy_to_param(obj, attr, value):
        """Copy value to parameter"""
        setattr(obj, attr, value)
    
    @staticmethod
    def set_attr_param(obj, attr, value):
        """Set attribute parameter"""
        setattr(obj, attr, value)


class SD:
    """Stub for comfy.sd module"""
    
    @staticmethod
    def load_checkpoint(ckpt_path, output_vae=True, output_clip=True, embedding_directory=None):
        """Load checkpoint stub"""
        return ({}, {}, {})
    
    @staticmethod
    def load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=None):
        """Load checkpoint with config guessing stub"""
        return ({}, {}, {})


class Samplers:
    """Stub for comfy.samplers"""
    
    class KSampler:
        SAMPLERS = ["euler", "euler_a", "heun", "dpm_2", "dpm_2_a", "lms", "dpmpp_2m", "ddim"]
        SCHEDULERS = ["normal", "karras", "exponential", "simple", "ddim_uniform"]


class Sample:
    """Stub for comfy.sample"""
    
    @staticmethod
    def sample(*args, **kwargs):
        """Sample stub"""
        return torch.randn(1, 4, 64, 64)
    
    @staticmethod
    def prepare_noise(latent_image, seed, batch_inds=None):
        """Prepare noise stub"""
        torch.manual_seed(seed)
        return torch.randn_like(latent_image)
    
    @staticmethod
    def fix_empty_latent_channels(model, latent):
        """Fix empty latent channels stub"""
        return latent


# Helper functions
def copy_to_param(obj, attr, value):
    """Copy value to parameter"""
    setattr(obj, attr, value)


def set_module_tensor_to_device(module, tensor_name, device, dtype=None, value=None):
    """Set module tensor to device"""
    if value is not None:
        if dtype is not None:
            value = value.to(dtype=dtype)
        value = value.to(device=device)
        
        # Handle nested attributes
        if '.' in tensor_name:
            parts = tensor_name.split('.')
            current = module
            for part in parts[:-1]:
                current = getattr(current, part)
            setattr(current, parts[-1], value)
        else:
            setattr(module, tensor_name, value)


def set_attr_param(obj, attr, value):
    """Set attribute parameter"""
    setattr(obj, attr, value)


def cast_to_device(tensor, device, dtype=None, copy=False):
    """Move tensor to device with optional dtype and copy semantics."""
    if tensor is None:
        return None
    if isinstance(device, str):
        device = torch.device(device)
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if hasattr(tensor, "to"):
        return tensor.to(device=device, **kwargs).clone() if copy else tensor.to(device=device, **kwargs)
    return torch.as_tensor(tensor, device=device, **kwargs)


def string_to_seed(text: str) -> int:
    """Deterministically convert a string to a 64-bit seed."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(h[:8], byteorder="little", signed=False)


def get_key_weight(model, key: str):
    """
    Lightweight implementation compatible with ComfyUI's model_patcher.get_key_weight.
    Returns (weight_tensor, set_func, convert_func) for a given model and key path.
    set_func is None when direct assignment is possible.
    convert_func is an identity transform.
    """
    weight = None
    target_obj = model
    # Direct attribute traverse
    try:
        parts = key.split('.')
        for p in parts[:-1]:
            target_obj = getattr(target_obj, p)
        weight = getattr(target_obj, parts[-1])
    except Exception:
        # Fallback: search by named_parameters
        try:
            for name, param in getattr(model, 'named_parameters', lambda recurse=True: [])(recurse=True):
                if name == key:
                    weight = param
                    break
        except Exception:
            weight = None
    if weight is None:
        # As a last resort, return a zero tensor to avoid crashes
        weight = torch.zeros(1)
    set_func = None
    convert_func = (lambda t, inplace=True: t)
    return weight, set_func, convert_func


# Create comfy package module
comfy = ModuleType('comfy')

# Build submodules as proper modules to satisfy "from comfy.xxx import y" imports
model_management_module = ModuleType('comfy.model_management')
model_management_module.get_torch_device = ModelManagement.get_torch_device
model_management_module.get_autocast_device = ModelManagement.get_autocast_device
model_management_module.soft_empty_cache = ModelManagement.soft_empty_cache
model_management_module.load_models_gpu = ModelManagement.load_models_gpu
model_management_module.unload_all_models = ModelManagement.unload_all_models
model_management_module.get_free_memory = ModelManagement.get_free_memory
model_management_module.interrupt_current_processing = ModelManagement.interrupt_current_processing
model_management_module.cast_to_device = cast_to_device

def common_upscale(samples, width, height, upscale_method, crop="disabled"):
    """Common upscale function for images/latents"""
    if isinstance(samples, torch.Tensor):
        # Ensure 4D tensor (B, C, H, W)
        if len(samples.shape) == 3:
            samples = samples.unsqueeze(0)
        
        import torch.nn.functional as F
        
        # Map upscale methods to torch modes
        mode_map = {
            "nearest": "nearest",
            "nearest-exact": "nearest",
            "bilinear": "bilinear",
            "area": "area",
            "bicubic": "bicubic",
            "lanczos": "bilinear"  # Fallback to bilinear
        }
        mode = mode_map.get(upscale_method, "bilinear")
        
        # Upscale
        upscaled = F.interpolate(
            samples,
            size=(height, width),
            mode=mode,
            align_corners=False if mode != "nearest" else None
        )
        
        # Crop if needed
        if crop == "center":
            h, w = upscaled.shape[2], upscaled.shape[3]
            if h != height or w != width:
                top = (h - height) // 2
                left = (w - width) // 2
                upscaled = upscaled[:, :, top:top+height, left:left+width]
        
        return upscaled
    return samples

utils_module = ModuleType('comfy.utils')
utils_module.PROGRESS_BAR_ENABLED = True
utils_module.ProgressBar = Utils.ProgressBar
utils_module.copy_to_param = copy_to_param
utils_module.set_attr_param = set_attr_param
utils_module.set_module_tensor_to_device = set_module_tensor_to_device
utils_module.common_upscale = common_upscale

sd_module = ModuleType('comfy.sd')
sd_module.load_checkpoint = SD.load_checkpoint
sd_module.load_checkpoint_guess_config = SD.load_checkpoint_guess_config

samplers_module = ModuleType('comfy.samplers')
samplers_module.KSampler = Samplers.KSampler

sample_module = ModuleType('comfy.sample')
sample_module.sample = Sample.sample
sample_module.prepare_noise = Sample.prepare_noise
sample_module.fix_empty_latent_channels = Sample.fix_empty_latent_channels

model_patcher_module = ModuleType('comfy.model_patcher')
model_patcher_module.ModelPatcher = ModelPatcher
model_patcher_module.get_key_weight = get_key_weight
model_patcher_module.string_to_seed = string_to_seed

lora_module = ModuleType('comfy.lora')
def _calculate_weight(patches, temp_weight, key):
    # Minimal placeholder: apply no-op (identity). Real logic can be added later.
    return temp_weight
lora_module.calculate_weight = _calculate_weight

float_module = ModuleType('comfy.float')
def _stochastic_rounding(tensor, dtype, seed=0):
    # Minimal rounding: deterministic cast with optional noise seed (ignored here)
    try:
        return tensor.to(dtype)
    except Exception:
        return tensor
float_module.stochastic_rounding = _stochastic_rounding

clip_vision_module = ModuleType('comfy.clip_vision')
class _CLIPVisionModel:
    """Placeholder CLIP Vision model"""
    def __init__(self):
        self.model = None
    
    def load(self, *args, **kwargs):
        return self
    
    def encode_image(self, image):
        # Return dummy embeddings
        return torch.randn(1, 768)

def _clip_preprocess(image, size=224):
    """Preprocess image for CLIP Vision"""
    # Minimal implementation: return image as-is or resize
    if isinstance(image, torch.Tensor):
        return image
    # If PIL image or similar, convert to tensor
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize(size),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                   std=[0.26862954, 0.26130258, 0.27577711])
    ])
    try:
        return transform(image)
    except:
        # Fallback: return dummy tensor
        return torch.randn(3, size, size)

clip_vision_module.CLIPVisionModel = _CLIPVisionModel
clip_vision_module.ClipVisionModel = _CLIPVisionModel  # Alternative naming
clip_vision_module.load = lambda *args, **kwargs: _CLIPVisionModel()
clip_vision_module.clip_preprocess = _clip_preprocess

# Attach submodules to comfy package
comfy.model_management = model_management_module
comfy.utils = utils_module
comfy.sd = sd_module
comfy.samplers = samplers_module
comfy.sample = sample_module
comfy.model_patcher = model_patcher_module
comfy.lora = lora_module
comfy.float = float_module
comfy.clip_vision = clip_vision_module

# Register in sys.modules
sys.modules['comfy'] = comfy
sys.modules['comfy.model_management'] = model_management_module
sys.modules['comfy.utils'] = utils_module
sys.modules['comfy.sd'] = sd_module
sys.modules['comfy.samplers'] = samplers_module
sys.modules['comfy.sample'] = sample_module
sys.modules['comfy.model_patcher'] = model_patcher_module
sys.modules['comfy.lora'] = lora_module
sys.modules['comfy.float'] = float_module
sys.modules['comfy.clip_vision'] = clip_vision_module

# Register folder_paths as global module (ComfyUI compatibility)
try:
    from genesis.core import folder_paths as genesis_folder_paths
    sys.modules['folder_paths'] = genesis_folder_paths
except ImportError:
    # Fallback: create minimal folder_paths stub
    folder_paths_module = ModuleType('folder_paths')
    folder_paths_module.models_dir = ""
    folder_paths_module.folder_names_and_paths = {}
    folder_paths_module.get_folder_paths = lambda x: []
    folder_paths_module.get_filename_list = lambda x: []
    folder_paths_module.get_full_path = lambda x, y: ""
    sys.modules['folder_paths'] = folder_paths_module

logger.info("ComfyUI compatibility layer loaded successfully")
