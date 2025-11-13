"""
Genesis Folder Paths Configuration
Model directory structure management
Author: eddy
"""

import os
from typing import Dict, List, Set, Tuple

supported_pt_extensions: Set[str] = {
    '.ckpt', '.pt', '.pt2', '.bin', '.pth', '.safetensors', '.pkl', '.sft'
}

base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
models_dir = os.path.join(base_path, "models")

folder_names_and_paths: Dict[str, Tuple[List[str], Set[str]]] = {}

folder_names_and_paths["checkpoints"] = (
    [os.path.join(models_dir, "checkpoints")], 
    supported_pt_extensions
)

folder_names_and_paths["configs"] = (
    [os.path.join(models_dir, "configs")], 
    {".yaml", ".json"}
)

folder_names_and_paths["loras"] = (
    [os.path.join(models_dir, "loras")], 
    supported_pt_extensions
)

folder_names_and_paths["vae"] = (
    [os.path.join(models_dir, "vae")], 
    supported_pt_extensions
)

folder_names_and_paths["text_encoders"] = (
    [os.path.join(models_dir, "text_encoders"), os.path.join(models_dir, "clip")], 
    supported_pt_extensions
)

folder_names_and_paths["clip"] = (
    [os.path.join(models_dir, "clip"), os.path.join(models_dir, "text_encoders")], 
    supported_pt_extensions
)

folder_names_and_paths["diffusion_models"] = (
    [os.path.join(models_dir, "unet"), os.path.join(models_dir, "diffusion_models")], 
    supported_pt_extensions
)

folder_names_and_paths["unet"] = (
    [os.path.join(models_dir, "unet"), os.path.join(models_dir, "diffusion_models")], 
    supported_pt_extensions
)

folder_names_and_paths["clip_vision"] = (
    [os.path.join(models_dir, "clip_vision")], 
    supported_pt_extensions
)

folder_names_and_paths["style_models"] = (
    [os.path.join(models_dir, "style_models")], 
    supported_pt_extensions
)

folder_names_and_paths["embeddings"] = (
    [os.path.join(models_dir, "embeddings")], 
    supported_pt_extensions
)

folder_names_and_paths["diffusers"] = (
    [os.path.join(models_dir, "diffusers")], 
    {"folder"}
)

folder_names_and_paths["vae_approx"] = (
    [os.path.join(models_dir, "vae_approx")], 
    supported_pt_extensions
)

folder_names_and_paths["controlnet"] = (
    [os.path.join(models_dir, "controlnet"), os.path.join(models_dir, "t2i_adapter")], 
    supported_pt_extensions
)

folder_names_and_paths["t2i_adapter"] = (
    [os.path.join(models_dir, "t2i_adapter"), os.path.join(models_dir, "controlnet")], 
    supported_pt_extensions
)

folder_names_and_paths["gligen"] = (
    [os.path.join(models_dir, "gligen")], 
    supported_pt_extensions
)

folder_names_and_paths["upscale_models"] = (
    [os.path.join(models_dir, "upscale_models")], 
    supported_pt_extensions
)

folder_names_and_paths["hypernetworks"] = (
    [os.path.join(models_dir, "hypernetworks")], 
    supported_pt_extensions
)

folder_names_and_paths["photomaker"] = (
    [os.path.join(models_dir, "photomaker")], 
    supported_pt_extensions
)

folder_names_and_paths["classifiers"] = (
    [os.path.join(models_dir, "classifiers")], 
    {""}
)

folder_names_and_paths["model_patches"] = (
    [os.path.join(models_dir, "model_patches")], 
    supported_pt_extensions
)

folder_names_and_paths["audio_encoders"] = (
    [os.path.join(models_dir, "audio_encoders")], 
    supported_pt_extensions
)

output_directory = os.path.join(base_path, "output")
temp_directory = os.path.join(base_path, "temp")
input_directory = os.path.join(base_path, "input")


def get_folder_paths(folder_name: str) -> List[str]:
    """
    Get folder paths for a given folder name
    
    Args:
        folder_name: Name of the folder type
        
    Returns:
        List of folder paths
    """
    if folder_name in folder_names_and_paths:
        return folder_names_and_paths[folder_name][0]
    return []


def get_supported_extensions(folder_name: str) -> Set[str]:
    """
    Get supported file extensions for a folder
    
    Args:
        folder_name: Name of the folder type
        
    Returns:
        Set of supported extensions
    """
    if folder_name in folder_names_and_paths:
        return folder_names_and_paths[folder_name][1]
    return set()


def get_filename_list(folder_name: str) -> List[str]:
    """
    Get list of files in a folder
    
    Args:
        folder_name: Name of the folder type
        
    Returns:
        List of filenames
    """
    folders = get_folder_paths(folder_name)
    extensions = get_supported_extensions(folder_name)
    
    files = []
    for folder in folders:
        if not os.path.exists(folder):
            continue
            
        for filename in os.listdir(folder):
            if "folder" in extensions:
                if os.path.isdir(os.path.join(folder, filename)):
                    files.append(filename)
            else:
                ext = os.path.splitext(filename)[1].lower()
                if ext in extensions:
                    files.append(filename)
    
    return sorted(files)


def get_full_path(folder_name: str, filename: str) -> str:
    """
    Get full path for a file
    
    Args:
        folder_name: Name of the folder type
        filename: Name of the file
        
    Returns:
        Full path to the file
    """
    folders = get_folder_paths(folder_name)
    
    for folder in folders:
        full_path = os.path.join(folder, filename)
        if os.path.exists(full_path):
            return full_path
    
    if folders:
        return os.path.join(folders[0], filename)
    
    return filename


def ensure_directories():
    """Create all model directories if they don't exist"""
    for folder_name, (paths, _) in folder_names_and_paths.items():
        for path in paths:
            os.makedirs(path, exist_ok=True)
    
    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(temp_directory, exist_ok=True)
    os.makedirs(input_directory, exist_ok=True)


def get_full_path_or_raise(folder_name: str, filename: str) -> str:
    """
    Get full path for a file, raising an exception if not found

    Args:
        folder_name: Name of the folder type
        filename: Name of the file

    Returns:
        Full path to the file

    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    folders = get_folder_paths(folder_name)

    for folder in folders:
        full_path = os.path.join(folder, filename)
        if os.path.exists(full_path):
            return full_path

    # Check models directory directly as fallback
    models_path = os.path.join("e:\\chai fream\\models", folder_name, filename)
    if os.path.exists(models_path):
        return models_path

    # Final fallback - check diffusion_models directly
    diffusion_path = os.path.join("e:\\chai fream\\models\\diffusion_models", filename)
    if os.path.exists(diffusion_path):
        return diffusion_path

    raise FileNotFoundError(f"Could not find {filename} in {folder_name} folders: {folders}")


def get_directory_structure() -> Dict[str, List[str]]:
    """
    Get the complete directory structure

    Returns:
        Dictionary mapping folder names to their paths
    """
    return {
        name: paths for name, (paths, _) in folder_names_and_paths.items()
    }


ensure_directories()
