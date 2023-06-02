#%%

import traceback
from pathlib import Path
import pandas as pd

import tensorflow as tf
from dlbd.evaluation import predictions
from dlbd.models import DLBD
from mouffet import common_utils
from mouffet.options.model_options import ModelOptions
from mouffet.utils.file import ensure_path_exists
from mouffet.utils.model_handler import ModelHandler

# gpus = tf.config.experimental.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(gpus[0], True)


dirs = {
    "raw": Path(
        "/mnt/win/UMoncton/Doctorat/data/dl_training/raw/predictions_raw/events_files"
    ),
    "extracts": Path(
        "/mnt/win/UMoncton/Doctorat/data/dl_training/raw/predictions_raw/extracts"
    ),
}


dest_root = Path("results/predictions")


model_opts = ModelOptions(
    {
        "model_dir": "../resources/models/",
        "name": "DLBD_v2",
        "model_id": "DLBD_v2",
        "class": DLBD,
        "batch_size": 64,
        "spectrogram_overlap": 0.75,
        "inference": True,
        "random_start": False,
        "ignore_parent_path": True,
        "input_height": 32,
        "class_size": "smallest",
        "pixels_per_sec": 100,
    }
)

overwrite = True


model = ModelHandler.load_model(model_opts)


spec_opts = {"n_fft": 512, "n_mels": 32, "sample_rate": "original", "to_db": False}

infos_res = []

for audio_type, root_dir in dirs.items():
    try:
        dest_path = (
            dest_root
            / f"{audio_type}_predictions_overlap-{model_opts['spectrogram_overlap']}.feather"
        )
        if not dest_path.exists():
            wav_list = list(root_dir.glob("*.WAV"))
            preds, infos = predictions.classify_elements(wav_list, model, spec_opts)
            preds.reset_index().to_feather(ensure_path_exists(dest_path, is_file=True))
            infos_res.append(infos)
            tmp_stats = pd.DataFrame([infos])
            tmp_stats.to_csv(
                dest_root / f"{audio_type}_predictions_stats.csv", index=False
            )
    except Exception:
        common_utils.print_error(traceback.format_exc())

    classification_stats = pd.DataFrame(infos_res)
    classification_stats.to_csv(dest_root / "global_stats.csv", index=False)

# %%
