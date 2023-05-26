#%%

from pathlib import Path

import numpy as np
import pandas as pd
from dlbd.data.audio_data_handler import AudioDataHandler
from dlbd.evaluation.song_detector_evaluation_handler import (
    SongDetectorEvaluationHandler,
)
from mouffet import common_utils, config_utils, file_utils
from plotnine import ggplot, aes, geom_point
from sklearn import metrics

pr_path = Path(
    "/mnt/win/UMoncton/Doctorat/dev/phenol1/final/results/PR_curves/PR_curves.feather"
)

method = "direct"

overwrite = False

if pr_path.exists() and not overwrite:
    pr_df = pd.read_feather(pr_path)
else:
    models_dir = "/mnt/win/UMoncton/Doctorat/dev/phenol1/resources/models"

    evaluation_config_path = f"config/{method}_pr_curve_evaluation_config.yaml"

    evaluation_config = file_utils.load_config(evaluation_config_path)

    evaluation_config["models_list_dir"] = models_dir

    print(evaluation_config)

    evaluation_config = config_utils.get_models_conf(
        evaluation_config,
    )

    evaluator = SongDetectorEvaluationHandler(
        opts=evaluation_config, dh_class=AudioDataHandler
    )
    stats = evaluator.evaluate()
    pr_df = stats["stats"]


#%%


def which(exp):
    return list(np.where(exp)[0])


def ensure_monotonous(df):
    df = df.reset_index(drop=True)
    try:
        outsiders = which(np.diff(df.recall) > 0)
        if len(outsiders) > 0:
            idx = outsiders  # [x + 1 if x > 0 else x for x in outsiders]
            df = df.drop(idx)
            return ensure_monotonous(df)
    except Exception:
        print(df)

    return df


def prepare_pr_curve(df):
    if isinstance(df.name, tuple):
        db, end_tresh = df.name
    else:
        db = df.name
        end_tresh = "N/A"
    tmp_pr_df = df.dropna(subset=["precision", "recall"])
    tmp_pr_df = pd.concat(
        [tmp_pr_df, pd.DataFrame([{"precision": 1, "recall": 0}])], ignore_index=True
    )
    tmp_pr_df = (
        tmp_pr_df.loc[:, ["precision", "recall"]]
        .sort_values("precision")
        .reset_index(drop=True)
    )
    tmp_pr_df = ensure_monotonous(tmp_pr_df)

    common_utils.print_warning(
        f"Metrics for database {db} and end threshold {end_tresh}: "
        + f"AUC: {round(metrics.auc(tmp_pr_df.recall, tmp_pr_df.precision), 3)}; "
        + f"AP: {round(-np.sum(np.diff(tmp_pr_df.recall) * np.array(tmp_pr_df.precision)[:-1]), 3)}"
    )
    return tmp_pr_df


groupby = {"standard": ["database", "end_threshold"], "direct": ["database"]}

for method in pr_df.evaluator.unique():
    common_utils.print_info(f"Processing method {method}")
    tmp_meth = pr_df.loc[pr_df.evaluator == method]

    pr_curves_df = (
        tmp_meth.groupby(groupby[method]).apply(prepare_pr_curve).reset_index()
    )
    if method == "standard":
        pr_curves_df.end_threshold = pr_curves_df.end_threshold.astype("category")

        pr_curves = (
            ggplot(
                pr_curves_df,
                aes(x="precision", y="recall", shape="end_threshold", color="database"),
            )
            + geom_point()
        )
    else:
        pr_curves = (
            ggplot(
                pr_curves_df,
                aes(x="precision", y="recall", color="database"),
            )
            + geom_point()
        )

    print(pr_curves)

    # pr_df = pr_df.loc[pr_df.recall < 0.998]
    # pr_df = pr_df.append([{"precision": 1, "recall": 0}], ignore_index=True)
    # pr_df = pr_df.loc[:, ["precision", "recall"]].sort_values("precision")


# print(metrics.auc(pr_df.recall, pr_df.precision))

# print(pr_df.precision.mean())

# ap = -np.sum(np.diff(pr_df.recall) * np.array(pr_df.precision)[:-1])
# print(ap)


# %%
