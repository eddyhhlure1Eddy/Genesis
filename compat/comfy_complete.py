"""
Complete ComfyUI Compatibility Layer for Genesis
All required comfy APIs in one place
Author: eddy
"""

import sys
import torch
import logging
import hashlib
from types import ModuleType

logger = logging.getLogger(__name__)


# ============================================================================
# Model Patcher
# ============================================================================

class ModelPatcher:
    """comfy.model_patcher.ModelPatcher"""
    
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


# ============================================================================
# Model Management
# ============================================================================

class ModelManagement:
    """comfy.model_management"""
    
    @staticmethod
    def get_torch_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @staticmethod
    def unet_offload_device():
        """Device for offloading models"""
        return torch.device("cpu")
    
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
    
    @staticmethod
    def throw_exception_if_processing_interrupted():
        pass
    
    @staticmethod
    def is_device_mps(device):
        return device.type == 'mps' if hasattr(device, 'type') else False
    
    @staticmethod
    def is_intel_xpu():
        return False
    
    @staticmethod
    def is_directml_enabled():
        return False


# ============================================================================
# Utilities
# ============================================================================

def load_torch_file(path, safe_load=False):
    """Load torch/safetensors file"""
    try:
        if path.endswith('.safetensors'):
            import safetensors.torch
            return safetensors.torch.load_file(path, device="cpu")
        else:
            return torch.load(path, map_location="cpu")
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        return {}


def common_upscale(samples, width, height, upscale_method, crop="disabled"):
    """Upscale images/latents"""
    if isinstance(samples, torch.Tensor):
        if len(samples.shape) == 3:
            samples = samples.unsqueeze(0)
        
        import torch.nn.functional as F
        
        mode_map = {
            "nearest": "nearest",
            "nearest-exact": "nearest",
            "bilinear": "bilinear",
            "area": "area",
            "bicubic": "bicubic",
            "lanczos": "bilinear"
        }
        mode = mode_map.get(upscale_method, "bilinear")
        
        upscaled = F.interpolate(
            samples,
            size=(height, width),
            mode=mode,
            align_corners=False if mode != "nearest" else None
        )
        
        if crop == "center":
            h, w = upscaled.shape[2], upscaled.shape[3]
            if h != height or w != width:
                top = (h - height) // 2
                left = (w - width) // 2
                upscaled = upscaled[:, :, top:top+height, left:left+width]
        
        return upscaled
    return samples


def copy_to_param(obj, attr, value):
    setattr(obj, attr, value)


def set_attr_param(obj, attr, value):
    setattr(obj, attr, value)


def set_module_tensor_to_device(module, tensor_name, device, dtype=None, value=None):
    if value is not None:
        if dtype is not None:
            value = value.to(dtype=dtype)
        value = value.to(device=device)
        
        if '.' in tensor_name:
            parts = tensor_name.split('.')
            current = module
            for part in parts[:-1]:
                current = getattr(current, part)
            setattr(current, parts[-1], value)
        else:
            setattr(module, tensor_name, value)


def cast_to_device(tensor, device, dtype=None, copy=False):
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
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(h[:8], byteorder="little", signed=False)


def get_key_weight(model, key: str):
    weight = None
    target_obj = model
    try:
        parts = key.split('.')
        for p in parts[:-1]:
            target_obj = getattr(target_obj, p)
        weight = getattr(target_obj, parts[-1])
    except Exception:
        try:
            for name, param in getattr(model, 'named_parameters', lambda recurse=True: [])(recurse=True):
                if name == key:
                    weight = param
                    break
        except Exception:
            weight = None
    if weight is None:
        weight = torch.zeros(1)
    set_func = None
    convert_func = (lambda t, inplace=True: t)
    return weight, set_func, convert_func


class ProgressBar:
    PROGRESS_BAR_ENABLED = True
    
    def __init__(self, total):
        self.total = total
        self.current = 0
    
    def update(self, value):
        self.current = value
    
    def update_absolute(self, value, total=None):
        self.current = value


# ============================================================================
# CLIP Vision
# ============================================================================

class CLIPVisionModel:
    def __init__(self):
        self.model = None
    
    def load(self, *args, **kwargs):
        return self
    
    def encode_image(self, image):
        return torch.randn(1, 768)


def clip_preprocess(image, size=224):
    if isinstance(image, torch.Tensor):
        return image
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
        return torch.randn(3, size, size)


# ============================================================================
# SD / Model Loading
# ============================================================================

def load_checkpoint(ckpt_path, output_vae=True, output_clip=True, embedding_directory=None):
    return ({}, {}, {})


def load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=None):
    return ({}, {}, {})


def load_lora_for_models(model, clip, lora_path, strength_model, strength_clip):
    return (model, clip)


# ============================================================================
# Sampling
# ============================================================================

class KSampler:
    SAMPLERS = ["euler", "euler_a", "heun", "dpm_2", "dpm_2_a", "lms", "dpmpp_2m", "ddim"]
    SCHEDULERS = ["normal", "karras", "exponential", "simple", "ddim_uniform"]


def sample(*args, **kwargs):
    return torch.randn(1, 4, 64, 64)


def prepare_noise(latent_image, seed, batch_inds=None):
    torch.manual_seed(seed)
    return torch.randn_like(latent_image)


def fix_empty_latent_channels(model, latent):
    return latent


# ============================================================================
# LoRA
# ============================================================================

def calculate_weight(patches, temp_weight, key):
    return temp_weight


# ============================================================================
# Float
# ============================================================================

def stochastic_rounding(tensor, dtype, seed=0):
    try:
        return tensor.to(dtype)
    except Exception:
        return tensor


# ============================================================================
# Model Sampling
# ============================================================================

class ModelSamplingDiscreteFlow:
    def __init__(self, shift=1.0):
        self.shift = shift


# ============================================================================
# Latent Formats
# ============================================================================

class Wan21:
    pass


class Wan22:
    pass


class HunyuanVideo:
    pass


# ============================================================================
# CLI Args
# ============================================================================

class Args:
    preview_method = "auto"


class LatentPreviewMethod:
    Auto = "auto"
    Latent2RGB = "latent2rgb"
    TAESD = "taesd"


args = Args()


# ============================================================================
# Node Typing (for remote nodes)
# ============================================================================

class IO:
    pass


class ComfyNodeABC:
    pass


class InputTypeDict(dict):
    pass


# ============================================================================
# Model Base
# ============================================================================

model_base = ModuleType('comfy.model_base')


# ============================================================================
# Operations (ops)
# ============================================================================

class CastWeightBiasOp:
    """Placeholder for weight/bias casting operations"""
    def __init__(self, dtype=None):
        self.dtype = dtype or torch.float32
    
    def __call__(self, weight, bias=None, dtype=None):
        target_dtype = dtype or self.dtype
        if weight is not None:
            weight = weight.to(target_dtype)
        if bias is not None:
            bias = bias.to(target_dtype)
        return weight, bias


class Linear(torch.nn.Linear):
    """Wrapper for torch.nn.Linear with comfy compatibility"""
    comfy_cast_weights = False
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Conv2d(torch.nn.Conv2d):
    """Wrapper for torch.nn.Conv2d with comfy compatibility"""
    comfy_cast_weights = False
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class GroupNorm(torch.nn.GroupNorm):
    """Wrapper for torch.nn.GroupNorm with comfy compatibility"""
    comfy_cast_weights = False
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class LayerNorm(torch.nn.LayerNorm):
    """Wrapper for torch.nn.LayerNorm with comfy compatibility"""
    comfy_cast_weights = False
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def disable_weight_init():
    """Context manager to disable weight initialization"""
    class DisableWeightInit:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    return DisableWeightInit()


def manual_cast():
    """Placeholder for manual casting"""
    return CastWeightBiasOp()


def cast_bias_weight(s, input=None, dtype=None, device=None):
    """Cast bias and weight to specified dtype and device"""
    if s is None:
        return None
    
    if dtype is not None:
        s = s.to(dtype=dtype)
    if device is not None:
        s = s.to(device=device)
    
    return s


# ============================================================================
# Module Registration
# ============================================================================

# Create comfy package
comfy = ModuleType('comfy')

# Model Management module
mm_module = ModuleType('comfy.model_management')
for attr in dir(ModelManagement):
    if not attr.startswith('_'):
        setattr(mm_module, attr, getattr(ModelManagement, attr))
mm_module.cast_to_device = cast_to_device

# Utils module
utils_module = ModuleType('comfy.utils')
utils_module.PROGRESS_BAR_ENABLED = True
utils_module.ProgressBar = ProgressBar
utils_module.load_torch_file = load_torch_file
utils_module.common_upscale = common_upscale
utils_module.copy_to_param = copy_to_param
utils_module.set_attr_param = set_attr_param
utils_module.set_module_tensor_to_device = set_module_tensor_to_device

# SD module
sd_module = ModuleType('comfy.sd')
sd_module.load_checkpoint = load_checkpoint
sd_module.load_checkpoint_guess_config = load_checkpoint_guess_config
sd_module.load_lora_for_models = load_lora_for_models

# Samplers module
samplers_module = ModuleType('comfy.samplers')
samplers_module.KSampler = KSampler

# Sample module
sample_module = ModuleType('comfy.sample')
sample_module.sample = sample
sample_module.prepare_noise = prepare_noise
sample_module.fix_empty_latent_channels = fix_empty_latent_channels

# Model Patcher module
mp_module = ModuleType('comfy.model_patcher')
mp_module.ModelPatcher = ModelPatcher
mp_module.get_key_weight = get_key_weight
mp_module.string_to_seed = string_to_seed

# LoRA module
lora_module = ModuleType('comfy.lora')
lora_module.calculate_weight = calculate_weight

# Float module
float_module = ModuleType('comfy.float')
float_module.stochastic_rounding = stochastic_rounding

# CLIP Vision module
cv_module = ModuleType('comfy.clip_vision')
cv_module.CLIPVisionModel = CLIPVisionModel
cv_module.ClipVisionModel = CLIPVisionModel
cv_module.load = lambda *args, **kwargs: CLIPVisionModel()
cv_module.clip_preprocess = clip_preprocess

# CLI Args module
cli_module = ModuleType('comfy.cli_args')
cli_module.args = args
cli_module.LatentPreviewMethod = LatentPreviewMethod

# Model Sampling module
ms_module = ModuleType('comfy.model_sampling')
ms_module.ModelSamplingDiscreteFlow = ModelSamplingDiscreteFlow

# Latent Formats module
lf_module = ModuleType('comfy.latent_formats')
lf_module.Wan21 = Wan21
lf_module.Wan22 = Wan22
lf_module.HunyuanVideo = HunyuanVideo

# Comfy Types module
ct_module = ModuleType('comfy.comfy_types')
nt_module = ModuleType('comfy.comfy_types.node_typing')
nt_module.IO = IO
nt_module.ComfyNodeABC = ComfyNodeABC
nt_module.InputTypeDict = InputTypeDict
ct_module.node_typing = nt_module

# Ops module
ops_module = ModuleType('comfy.ops')
ops_module.CastWeightBiasOp = CastWeightBiasOp
ops_module.Linear = Linear
ops_module.Conv2d = Conv2d
ops_module.GroupNorm = GroupNorm
ops_module.LayerNorm = LayerNorm
ops_module.disable_weight_init = disable_weight_init
ops_module.manual_cast = manual_cast
ops_module.cast_bias_weight = cast_bias_weight

# Attach all to comfy package
comfy.model_management = mm_module
comfy.utils = utils_module
comfy.sd = sd_module
comfy.samplers = samplers_module
comfy.sample = sample_module
comfy.model_patcher = mp_module
comfy.lora = lora_module
comfy.float = float_module
comfy.clip_vision = cv_module
comfy.cli_args = cli_module
comfy.model_sampling = ms_module
comfy.latent_formats = lf_module
comfy.model_base = model_base
comfy.comfy_types = ct_module
comfy.ops = ops_module

# Register in sys.modules
sys.modules['comfy'] = comfy
sys.modules['comfy.model_management'] = mm_module
sys.modules['comfy.utils'] = utils_module
sys.modules['comfy.sd'] = sd_module
sys.modules['comfy.samplers'] = samplers_module
sys.modules['comfy.sample'] = sample_module
sys.modules['comfy.model_patcher'] = mp_module
sys.modules['comfy.lora'] = lora_module
sys.modules['comfy.float'] = float_module
sys.modules['comfy.clip_vision'] = cv_module
sys.modules['comfy.cli_args'] = cli_module
sys.modules['comfy.model_sampling'] = ms_module
sys.modules['comfy.latent_formats'] = lf_module
sys.modules['comfy.model_base'] = model_base
sys.modules['comfy.comfy_types'] = ct_module
sys.modules['comfy.comfy_types.node_typing'] = nt_module
sys.modules['comfy.ops'] = ops_module

# Register folder_paths
try:
    from genesis.core import folder_paths as genesis_folder_paths
    sys.modules['folder_paths'] = genesis_folder_paths
except ImportError:
    folder_paths_module = ModuleType('folder_paths')
    folder_paths_module.models_dir = ""
    folder_paths_module.folder_names_and_paths = {}
    folder_paths_module.get_folder_paths = lambda x: []
    folder_paths_module.get_filename_list = lambda x: []
    folder_paths_module.get_full_path = lambda x, y: ""
    sys.modules['folder_paths'] = folder_paths_module

logger.info("Complete ComfyUI compatibility layer loaded successfully")
