#%%

from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from dlbd.evaluation import predictions
from dlbd.models import DLBD
from mouffet.options.model_options import ModelOptions
from mouffet.utils import file_utils
from mouffet.utils.model_handler import ModelHandler

gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)


model_opts = ModelOptions(
    {
        "model_dir": "resources/models/",
        "name": "DLBD_v2",
        "model_id": "DLBD_v2",
        "class": DLBD,
        "batch_size": 64,
        "spectrogram_overlap": 0.95,
        "inference": True,
        "random_start": False,
        "ignore_parent_path": True,
    }
)

model = ModelHandler.load_model(model_opts)


print(model.opts.opts)

spec_opts = {"n_fft": 512, "n_mels": 32, "sample_rate": "original", "to_db": False}


file_paths = [
    Path("resources/audio/original.wav"),
    Path("resources/audio/extract.wav"),
    Path("resources/audio/denoised.wav"),
    Path("resources/audio/original3.wav"),
    Path("resources/audio/extract3.wav"),
    Path("resources/audio/denoised3.wav"),
]

dest_dir = Path("results/check_preds")
i = 4

for file_path in file_paths:
    preds, info = predictions.classify_elements([file_path], model, spec_opts)
    preds.plot("time", "activity")
    plt.savefig(
        file_utils.ensure_path_exists(dest_dir) / f"{i}_{file_path.stem}_mod.png"
    )
    i = i + 1
