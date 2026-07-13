from pathlib import Path
import os

def get_subfolders_from_folder(folder_name):
    subfolders = [f.path + os.sep for f in os.scandir(folder_name) if f.is_dir()]
    return subfolders