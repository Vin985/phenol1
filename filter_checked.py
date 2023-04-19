#%%

import pandas as pd
from pathlib import Path
from plotnine import *
from pyutils.file_utils import ensure_path_exists
import shutil

import configparser


src_path = Path("/home/vin/Doctorat/data/dl_training/raw/ArcticChecked/")
dest_path = src_path / "filtered"


def load_files_done(path):
    files_done = []
    local_conf = path / "config.conf"
    if local_conf.exists():
        config = configparser.ConfigParser()
        config.read(local_conf)
        files_done = config["files"].get("files_done", [])
        if type(files_done) is str:
            files_done = files_done.split(",")
        files_done = [path / file_name for file_name in files_done]
    return files_done


def get_files_done(path, dest_dir, root=None):
    if root is None:
        root = path
    current_files_done = load_files_done(path)
    for file_path in path.iterdir():
        if file_path.is_dir():
            get_files_done(file_path, dest_dir, root=root)
        elif (
            file_path.is_file()
            and file_path.suffix.lower() in [".wav"]
            and file_path in current_files_done
        ):
            dest_file = Path(str(file_path).replace(str(root), str(dest_dir)))
            if not dest_file.exists():
                dest_file = ensure_path_exists(dest_file, is_file=True)
                print(f"Copying {file_path} to {dest_file}")
                shutil.copy(file_path, dest_file)
            tag_file = file_path.with_name(dest_file.stem + "-tags.csv")
            if tag_file.exists():
                dest_tag_file = Path(str(tag_file).replace(str(root), str(dest_dir)))
                if not dest_tag_file.exists():
                    print(f"Copying {tag_file} to {dest_tag_file}")
                    shutil.copy(tag_file, dest_tag_file)


get_files_done(src_path, dest_path)

# %%
