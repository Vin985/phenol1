#%%

from pathlib import Path

import mouffet.utils.file as file_utils
import numpy as np
import pandas as pd
from dlbd.data.audio_data_handler import AudioDataHandler
from dlbd.evaluation.song_detector_evaluation_handler import (
    SongDetectorEvaluationHandler,
)
from dlbd.utils import get_models_conf
from plotnine import *
from sklearn import metrics

pr_path = Path(
    "/mnt/win/UMoncton/Doctorat/dev/phenol1/final/results/PR_curves/PR_curves.feather"
)

overwrite = True

if pr_path.exists() and not overwrite:
    pr_df = pd.read_feather(pr_path)
else:
    models_dir = "/mnt/win/UMoncton/Doctorat/dev/phenol1/resources/models"

    evaluation_config_path = "config/standard_pr_curve_evaluation_config.yaml"

    evaluation_config = file_utils.load_config(evaluation_config_path)

    evaluation_config["models_list_dir"] = models_dir

    print(evaluation_config)

    evaluation_config = get_models_conf(
        evaluation_config,
    )

    evaluator = SongDetectorEvaluationHandler(
        opts=evaluation_config, dh_class=AudioDataHandler
    )
    stats = evaluator.evaluate()
    pr_df = stats["stats"]

pr_df = pr_df.loc[pr_df.recall < 0.998]
pr_df = pr_df.append([{"precision": 1, "recall": 0}], ignore_index=True)
pr_df = pr_df.loc[:, ["precision", "recall"]].sort_values("precision")


pr_curve = ggplot(pr_df, aes(x="precision", y="recall")) + geom_point()

print(metrics.auc(pr_df.recall, pr_df.precision))

print(pr_df.precision.mean())

ap = -np.sum(np.diff(pr_df.recall) * np.array(pr_df.precision)[:-1])
print(ap)


# %%
