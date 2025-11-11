"""
Genesis Pipeline - Workflow Builder
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class Node:
    """Node definition"""
    id: str
    type: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)


class Pipeline:
    """
    Workflow pipeline builder
    
    Provides simple API to build complex image generation workflows
    
    Example:
        >>> pipeline = Pipeline("my_workflow")
        >>> loader_id = pipeline.add_node("loader", "CheckpointLoader", checkpoint="model.safetensors")
        >>> sampler_id = pipeline.add_node("sampler", "KSampler", steps=20)
        >>> pipeline.connect(loader_id, "MODEL", sampler_id, "model")
    """
    
    def __init__(self, name: str = "pipeline"):
        """
        Initialize Pipeline
        
        Args:
            name: Pipeline name
        """
        self.name = name
        self.nodes: Dict[str, Node] = {}
        self.connections: List[Dict] = []
        self._node_counter = 0
        
    def add_node(
        self,
        node_type: str,
        name: Optional[str] = None,
        **inputs
    ) -> str:
        """
        Add node
        
        Args:
            node_type: Node type
            name: Node name (optional, auto-generated)
            **inputs: Node input parameters
            
        Returns:
            Node ID
        """
        # Generate node ID
        self._node_counter += 1
        node_id = f"{self._node_counter}"
        
        # Create node
        node_name = name or f"{node_type}_{node_id}"
        node = Node(
            id=node_id,
            type=node_type,
            inputs=inputs
        )
        
        self.nodes[node_id] = node
        return node_id
    
    def connect(
        self,
        source_node: str,
        source_output: str,
        target_node: str,
        target_input: str
    ):
        """
        Connect two nodes
        
        Args:
            source_node: Source node ID
            source_output: Source node output name
            target_node: Target node ID
            target_input: Target node input name
        """
        connection = {
            'source': {'node': source_node, 'output': source_output},
            'target': {'node': target_node, 'input': target_input}
        }
        
        self.connections.append(connection)
        
        # Update target node inputs
        if target_node in self.nodes:
            self.nodes[target_node].inputs[target_input] = {
                'from_node': source_node,
                'from_output': source_output
            }
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get node"""
        return self.nodes.get(node_id)
    
    def remove_node(self, node_id: str):
        """Remove node"""
        if node_id in self.nodes:
            del self.nodes[node_id]
            
        # Remove related connections
        self.connections = [
            conn for conn in self.connections
            if conn['source']['node'] != node_id and conn['target']['node'] != node_id
        ]
    
    def to_dict(self) -> Dict:
        """
        Convert to dictionary format
        
        Returns:
            Pipeline dictionary representation
        """
        return {
            'name': self.name,
            'nodes': {
                node_id: {
                    'id': node.id,
                    'type': node.type,
                    'inputs': node.inputs,
                    'outputs': node.outputs
                }
                for node_id, node in self.nodes.items()
            },
            'connections': self.connections
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Pipeline':
        """
        Create Pipeline from dictionary
        
        Args:
            data: Pipeline dictionary data
            
        Returns:
            Pipeline instance
        """
        pipeline = cls(name=data.get('name', 'pipeline'))
        
        # Restore nodes
        for node_id, node_data in data.get('nodes', {}).items():
            node = Node(
                id=node_data['id'],
                type=node_data['type'],
                inputs=node_data.get('inputs', {}),
                outputs=node_data.get('outputs', {})
            )
            pipeline.nodes[node_id] = node
            
        # Restore connections
        pipeline.connections = data.get('connections', [])
        
        # Update counter
        if pipeline.nodes:
            pipeline._node_counter = max(int(nid) for nid in pipeline.nodes.keys())
            
        return pipeline
    
    def validate(self) -> List[str]:
        """
        Validate Pipeline
        
        Returns:
            Error list, empty list means no errors
        """
        errors = []
        
        # Check connections
        for conn in self.connections:
            source_node = conn['source']['node']
            target_node = conn['target']['node']
            
            if source_node not in self.nodes:
                errors.append(f"Source node '{source_node}' not found")
                
            if target_node not in self.nodes:
                errors.append(f"Target node '{target_node}' not found")
                
        return errors
    
    def __repr__(self) -> str:
        return f"<Pipeline(name='{self.name}', nodes={len(self.nodes)}, connections={len(self.connections)})>"


class PipelineBuilder:
    """Pipeline builder helper class"""
    
    @staticmethod
    def text_to_image(
        checkpoint: str,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        cfg: float = 7.0,
        seed: int = -1
    ) -> Pipeline:
        """
        Create text-to-image Pipeline
        
        Args:
            checkpoint: Checkpoint name
            prompt: Positive prompt
            negative_prompt: Negative prompt
            width: Width
            height: Height
            steps: Sampling steps
            cfg: CFG scale
            seed: Random seed
            
        Returns:
            Pipeline instance
        """
        pipeline = Pipeline("text_to_image")
        
        # Load model
        loader = pipeline.add_node("CheckpointLoader", checkpoint=checkpoint)
        
        # Encode prompts
        clip_text = pipeline.add_node("CLIPTextEncode", text=prompt)
        clip_text_neg = pipeline.add_node("CLIPTextEncode", text=negative_prompt)
        
        # Empty latent
        empty_latent = pipeline.add_node("EmptyLatentImage", width=width, height=height, batch_size=1)
        
        # Sampling
        sampler = pipeline.add_node(
            "KSampler",
            steps=steps,
            cfg=cfg,
            seed=seed,
            sampler_name="euler",
            scheduler="normal"
        )
        
        # Decode
        decoder = pipeline.add_node("VAEDecode")
        
        # Save image
        save = pipeline.add_node("SaveImage", filename_prefix="genesis")
        
        # Connect nodes
        pipeline.connect(loader, "MODEL", sampler, "model")
        pipeline.connect(loader, "CLIP", clip_text, "clip")
        pipeline.connect(loader, "CLIP", clip_text_neg, "clip")
        pipeline.connect(loader, "VAE", decoder, "vae")
        
        pipeline.connect(clip_text, "CONDITIONING", sampler, "positive")
        pipeline.connect(clip_text_neg, "CONDITIONING", sampler, "negative")
        pipeline.connect(empty_latent, "LATENT", sampler, "latent_image")
        
        pipeline.connect(sampler, "LATENT", decoder, "samples")
        pipeline.connect(decoder, "IMAGE", save, "images")
        
        return pipeline
