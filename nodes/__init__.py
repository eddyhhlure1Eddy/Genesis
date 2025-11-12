"""
Genesis Native Nodes
ComfyUI-compatible node implementations

WARNING: Nodes are no longer globally imported!
Please use genesis.core.node_loader to dynamically load required modules

Example:
    from genesis.core import node_loader

    # Load WAN video module
    node_loader.load_module('wan_video')

    # Use nodes
    WanVideoSampler = node_loader.get_node_class('WanVideoSampler')

Author: eddy
"""

# No longer auto-import all modules
# from .loaders import *
# from .samplers import *
# from .conditioning import *
# from .latent import *
# from .image import *
# from .wan_video import *

# If you need manual on-demand import, you can do:
# from genesis.nodes.wan_video import WanVideoSampler

__all__ = []  # Export nothing, use node_loader for dynamic loading
