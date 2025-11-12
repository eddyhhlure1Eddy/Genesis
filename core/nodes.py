"""
Genesis Nodes - Node Definition System
Node Definition System
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class NodeInput:
    """Node input definition"""
    name: str
    type: str
    required: bool = True
    default: Any = None
    description: str = ""


@dataclass
class NodeOutput:
    """Node output definition"""
    name: str
    type: str
    description: str = ""


class BaseNode(ABC):
    """
    Base class for all nodes
    
    All custom nodes should inherit from this class
    """
    
    # Node metadata
    CATEGORY: str = "base"
    DISPLAY_NAME: str = "Base Node"
    DESCRIPTION: str = "Base node class"
    
    # Define inputs and outputs
    INPUTS: List[NodeInput] = []
    OUTPUTS: List[NodeOutput] = []
    
    def __init__(self, node_id: str, **params):
        """
        Initialize node
        
        Args:
            node_id: Unique node identifier
            **params: Node parameters
        """
        self.node_id = node_id
        self.params = params
        self.outputs_cache = {}
        
    @abstractmethod
    def execute(self, **inputs) -> Dict[str, Any]:
        """
        Execute node logic
        
        Args:
            **inputs: Input values
            
        Returns:
            Dictionary of output values
        """
        pass
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """
        Validate input values
        
        Args:
            inputs: Input dictionary
            
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        for input_def in self.INPUTS:
            if input_def.required and input_def.name not in inputs:
                errors.append(f"Required input '{input_def.name}' is missing")
                
        return errors
    
    @classmethod
    def get_node_info(cls) -> Dict[str, Any]:
        """Get node information"""
        return {
            'category': cls.CATEGORY,
            'display_name': cls.DISPLAY_NAME,
            'description': cls.DESCRIPTION,
            'inputs': [
                {
                    'name': inp.name,
                    'type': inp.type,
                    'required': inp.required,
                    'default': inp.default,
                    'description': inp.description
                }
                for inp in cls.INPUTS
            ],
            'outputs': [
                {
                    'name': out.name,
                    'type': out.type,
                    'description': out.description
                }
                for out in cls.OUTPUTS
            ]
        }


# ===== Built-in Node Types =====

class CheckpointLoaderNode(BaseNode):
    """Checkpoint loader node"""
    
    CATEGORY = "loaders"
    DISPLAY_NAME = "Load Checkpoint"
    DESCRIPTION = "Load a checkpoint model"
    
    INPUTS = [
        NodeInput("checkpoint", "STRING", required=True, description="Checkpoint filename")
    ]
    
    OUTPUTS = [
        NodeOutput("MODEL", "MODEL", "Loaded model"),
        NodeOutput("CLIP", "CLIP", "CLIP text encoder"),
        NodeOutput("VAE", "VAE", "VAE decoder")
    ]
    
    def execute(self, checkpoint: str) -> Dict[str, Any]:
        """Load checkpoint"""
        # TODO: Implement actual loading
        return {
            'MODEL': {'type': 'model', 'checkpoint': checkpoint},
            'CLIP': {'type': 'clip', 'checkpoint': checkpoint},
            'VAE': {'type': 'vae', 'checkpoint': checkpoint}
        }


class CLIPTextEncodeNode(BaseNode):
    """CLIP text encoder node"""
    
    CATEGORY = "conditioning"
    DISPLAY_NAME = "CLIP Text Encode"
    DESCRIPTION = "Encode text prompt using CLIP"
    
    INPUTS = [
        NodeInput("text", "STRING", required=True, description="Text prompt"),
        NodeInput("clip", "CLIP", required=True, description="CLIP model")
    ]
    
    OUTPUTS = [
        NodeOutput("CONDITIONING", "CONDITIONING", "Encoded conditioning")
    ]
    
    def execute(self, text: str, clip: Any) -> Dict[str, Any]:
        """Encode text"""
        # TODO: Implement actual encoding
        return {
            'CONDITIONING': {'text': text, 'encoded': True}
        }


class EmptyLatentImageNode(BaseNode):
    """Empty latent image node"""
    
    CATEGORY = "latent"
    DISPLAY_NAME = "Empty Latent Image"
    DESCRIPTION = "Create an empty latent image"
    
    INPUTS = [
        NodeInput("width", "INT", required=True, default=512, description="Image width"),
        NodeInput("height", "INT", required=True, default=512, description="Image height"),
        NodeInput("batch_size", "INT", required=False, default=1, description="Batch size")
    ]
    
    OUTPUTS = [
        NodeOutput("LATENT", "LATENT", "Empty latent image")
    ]
    
    def execute(self, width: int, height: int, batch_size: int = 1) -> Dict[str, Any]:
        """Create empty latent"""
        return {
            'LATENT': {
                'width': width,
                'height': height,
                'batch_size': batch_size
            }
        }


class KSamplerNode(BaseNode):
    """K-Sampler node"""
    
    CATEGORY = "sampling"
    DISPLAY_NAME = "KSampler"
    DESCRIPTION = "Sample using K-Diffusion samplers"
    
    INPUTS = [
        NodeInput("model", "MODEL", required=True),
        NodeInput("positive", "CONDITIONING", required=True),
        NodeInput("negative", "CONDITIONING", required=True),
        NodeInput("latent_image", "LATENT", required=True),
        NodeInput("seed", "INT", default=-1),
        NodeInput("steps", "INT", default=20),
        NodeInput("cfg", "FLOAT", default=7.0),
        NodeInput("sampler_name", "STRING", default="euler"),
        NodeInput("scheduler", "STRING", default="normal"),
        NodeInput("denoise", "FLOAT", default=1.0)
    ]
    
    OUTPUTS = [
        NodeOutput("LATENT", "LATENT", "Sampled latent")
    ]
    
    def execute(self, **inputs) -> Dict[str, Any]:
        """Execute sampling"""
        # TODO: Implement actual sampling
        return {
            'LATENT': {
                'sampled': True,
                'steps': inputs.get('steps', 20),
                'cfg': inputs.get('cfg', 7.0)
            }
        }


class VAEDecodeNode(BaseNode):
    """VAE decoder node"""
    
    CATEGORY = "latent"
    DISPLAY_NAME = "VAE Decode"
    DESCRIPTION = "Decode latent to image"
    
    INPUTS = [
        NodeInput("samples", "LATENT", required=True),
        NodeInput("vae", "VAE", required=True)
    ]
    
    OUTPUTS = [
        NodeOutput("IMAGE", "IMAGE", "Decoded image")
    ]
    
    def execute(self, samples: Any, vae: Any) -> Dict[str, Any]:
        """Decode latent"""
        # TODO: Implement actual decoding
        return {
            'IMAGE': {'decoded': True, 'from_latent': samples}
        }


class SaveImageNode(BaseNode):
    """Save image node"""
    
    CATEGORY = "image"
    DISPLAY_NAME = "Save Image"
    DESCRIPTION = "Save image to file"
    
    INPUTS = [
        NodeInput("images", "IMAGE", required=True),
        NodeInput("filename_prefix", "STRING", default="genesis")
    ]
    
    OUTPUTS = []
    
    def execute(self, images: Any, filename_prefix: str = "genesis") -> Dict[str, Any]:
        """Save image"""
        # TODO: Implement actual saving
        return {}


# ===== Node Registry =====

class NodeRegistry:
    """Node registry for managing available nodes"""

    def __init__(self, auto_register_builtin: bool = False):
        """
        Initialize node registry

        Args:
            auto_register_builtin: If True, automatically register built-in nodes.
                                  Default False for lazy loading.
        """
        self.nodes: Dict[str, type] = {}
        self._builtin_registered = False

        if auto_register_builtin:
            self._register_builtin_nodes()

    def _register_builtin_nodes(self):
        """Register built-in nodes (call explicitly when needed)"""
        if self._builtin_registered:
            return

        builtin_nodes = [
            CheckpointLoaderNode,
            CLIPTextEncodeNode,
            EmptyLatentImageNode,
            KSamplerNode,
            VAEDecodeNode,
            SaveImageNode
        ]

        for node_class in builtin_nodes:
            self.register(node_class)

        self._builtin_registered = True

    def ensure_builtin_registered(self):
        """Ensure built-in nodes are registered (lazy initialization)"""
        if not self._builtin_registered:
            self._register_builtin_nodes()
    
    def register(self, node_class: type):
        """
        Register a node class
        
        Args:
            node_class: Node class to register
        """
        if not issubclass(node_class, BaseNode):
            raise TypeError(f"{node_class} must inherit from BaseNode")
        
        node_name = node_class.__name__.replace('Node', '')
        self.nodes[node_name] = node_class
        
    def get(self, node_type: str, auto_register: bool = True) -> Optional[type]:
        """
        Get node class by type

        Args:
            node_type: Node type name
            auto_register: If True, auto-register built-in nodes if not found

        Returns:
            Node class or None
        """
        if auto_register and node_type not in self.nodes:
            self.ensure_builtin_registered()

        return self.nodes.get(node_type)

    def list_nodes(self, category: Optional[str] = None, include_builtin: bool = True) -> List[Dict[str, Any]]:
        """
        List all registered nodes

        Args:
            category: Filter by category
            include_builtin: If True, ensure built-in nodes are registered

        Returns:
            List of node information
        """
        if include_builtin:
            self.ensure_builtin_registered()

        nodes = []
        for name, node_class in self.nodes.items():
            info = node_class.get_node_info()
            info['name'] = name

            if category is None or info['category'] == category:
                nodes.append(info)

        return nodes
    
    def list_categories(self) -> List[str]:
        """List all node categories"""
        categories = set()
        for node_class in self.nodes.values():
            categories.add(node_class.CATEGORY)
        return sorted(list(categories))


# Global node registry (lazy loading - no auto-registration)
NODE_REGISTRY = NodeRegistry(auto_register_builtin=False)
