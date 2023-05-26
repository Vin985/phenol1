#%%

import pandas as pd
import patchworklib as pw
from dlbd.applications.phenology.phenology_evaluator import PhenologyEvaluator
from dlbd.data.audio_data_handler import AudioDataHandler
from dlbd.evaluation import EVALUATORS
from dlbd.evaluation.song_detector_evaluation_handler import (
    SongDetectorEvaluationHandler,
)
from dlbd.plots.distance import plot_norm_distance
from dlbd.plots.utils import format_date_short
from mouffet.utils import common_utils, config_utils, file_utils
from plotnine import *

EVALUATORS.register_evaluator(PhenologyEvaluator)

#%%


models_dir = "/mnt/win/UMoncton/Doctorat/dev/phenol1/resources/models"

evaluation_config_path = "config/paper_plots_config.yaml"

evaluation_config = file_utils.load_config(evaluation_config_path)

evaluation_config["models_list_dir"] = models_dir

print(evaluation_config)

evaluation_config = config_utils.get_models_conf(
    evaluation_config,
)

#%%

# * ARCTIC SUMMER

opts = {
    "evaluators": [
        {
            "type": "phenology",
            "method": "direct",
            "activity_threshold": 0.75,
            # "scenarios": {"remove_crows": [True, False]},
        }
    ],
    "databases": [{"name": "full_summer_final"}],
}

full_summer_opts = common_utils.deep_dict_update(evaluation_config, opts, copy=True)


evaluator = SongDetectorEvaluationHandler(
    opts=full_summer_opts, dh_class=AudioDataHandler
)

stats = evaluator.evaluate()

nrow = 0
scenario = stats["stats"].iloc[nrow]
full_summer_plt_data = {
    "df": stats["trends_df"][nrow]["trends_df"],
    "distance": scenario.eucl_distance,
    "distance_norm": scenario.eucl_distance_norm,
    "method": "direct",
}
full_summer_plt = plot_norm_distance(
    full_summer_plt_data,
    config_utils.load_options(scenario.evaluator_opts),
    "",
    ylab="Normalized total vocal activity\nper recording",
    title="",
)

full_summer_plt = (
    full_summer_plt
    + geom_line(size=1)
    + scale_color_manual(
        values=["#0571b0", "#f4a582"],
        labels=["Reference", "Model"],
    )
    # + ggtitle("")
    + scale_x_datetime(date_breaks="4 days", labels=format_date_short)
    # + ylab("Normalized total vocal activity\nper recording")
    + theme(
        axis_title=element_text(size=14, ha="center", va="center"),
        legend_title=element_blank(),
        legend_position=(0.9, 0.85),
        legend_direction="vertical",
        figure_size=(7, 5),
        legend_text=element_text(size=12),
        plot_margin=0.05,
    )
)
fs_plt = pw.load_ggplot(full_summer_plt)


#%%


# * ENAB

opts = {
    "evaluators": [
        {
            "type": "phenology",
            "method": "direct",
            "activity_threshold": 0.75,
            "scenarios": {"remove_crows": [False, True]},
        }
    ],
    "databases": [{"name": "ENAB_final"}],
}

enab_opts = common_utils.deep_dict_update(evaluation_config, opts, copy=True)


evaluator = SongDetectorEvaluationHandler(opts=enab_opts, dh_class=AudioDataHandler)

stats = evaluator.evaluate()

#%%

crow_row = 0
crows = stats["stats"].iloc[crow_row]
crows_df = stats["trends_df"][crow_row]["trends_df"].copy()
crows_df.type = crows_df.type.cat.rename_categories({"DLBD_v2": "crows"})

no_crow_row = 1
no_crows = stats["stats"].iloc[no_crow_row]
no_crows_df = stats["trends_df"][no_crow_row]["trends_df"]
no_crows_df.type = no_crows_df.type.cat.rename_categories({"ground_truth": "no_crows"})

enab_df = pd.concat(
    [crows_df, no_crows_df.loc[no_crows_df.type == "no_crows"]]
).reset_index(drop=True)
enab_df.type = enab_df.type.astype("category")


enab_plt = (
    ggplot(
        data=enab_df,
        mapping=aes(
            "date", "trend_norm", color="type", linetype="type"
        ),  # , alpha="type"),
    )
    + geom_line(
        size=1,
    )
    + scale_linetype_manual(values=["solid", "solid", "dashed"], guide=None)
    + scale_color_manual(
        values=["#f4a582", "#0571b0", "#0571b0"],
        labels=["Predictions", "Reference", "Reference without crows"],
    )
    # + scale_alpha_manual(
    #     values=[1, 0.7, 0.7],
    #     guide=None,
    # )
    + scale_x_datetime(date_breaks="30 minutes", labels=format_date_short)
    + xlab("Date")
    + ylab("Normalized total vocal activity\nper recording")
    + theme_classic()
    + theme(
        axis_title=element_text(size=14, ha="center", va="center"),
        axis_text_x=element_text(angle=45),
        legend_title=element_blank(),
        legend_position=(0.8, 0.3),
        legend_direction="vertical",
        figure_size=(7, 5),
        legend_text=element_text(size=12),
        plot_margin=0.05,
    )
)
e_plt = pw.load_ggplot(enab_plt)

#%%

final = fs_plt | e_plt

final.savefig("final_composed.png")
