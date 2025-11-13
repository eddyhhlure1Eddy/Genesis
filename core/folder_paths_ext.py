"""
Extension functions for folder_paths module
Provides ComfyUI compatibility functions
Author: eddy
"""

import os
import glob
from . import folder_paths


def add_model_folder_path(folder_name, paths):
    """Add model folder paths for ComfyUI compatibility"""
    if folder_name not in folder_paths.folder_names_and_paths:
        folder_paths.folder_names_and_paths[folder_name] = ([], set())

    current_paths, extensions = folder_paths.folder_names_and_paths[folder_name]

    if isinstance(paths, str):
        paths = [paths]

    for path in paths:
        if path not in current_paths:
            current_paths.append(path)


def get_folder_paths(folder_name):
    """Get folder paths for a given folder name"""
    if folder_name in folder_paths.folder_names_and_paths:
        return folder_paths.folder_names_and_paths[folder_name][0]
    return []


def get_filename_list(folder_name):
    """Get list of files in a folder"""
    paths = get_folder_paths(folder_name)
    files = []

    if folder_name in folder_paths.folder_names_and_paths:
        extensions = folder_paths.folder_names_and_paths[folder_name][1]
        for path in paths:
            if os.path.exists(path):
                for ext in extensions:
                    pattern = os.path.join(path, f"*{ext}")
                    files.extend([os.path.basename(f) for f in glob.glob(pattern)])

    return list(set(files))


def get_full_path(folder_name, filename):
    """Get full path for a file in a folder"""
    paths = get_folder_paths(folder_name)
    for path in paths:
        full_path = os.path.join(path, filename)
        if os.path.exists(full_path):
            return full_path

    # Fallback to first path
    if paths:
        return os.path.join(paths[0], filename)
    return None


# Inject these functions into the folder_paths module
folder_paths.add_model_folder_path = add_model_folder_path
folder_paths.get_folder_paths = get_folder_paths
folder_paths.get_filename_list = get_filename_list
folder_paths.get_full_path = get_full_path