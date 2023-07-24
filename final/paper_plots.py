#%%
from pathlib import Path
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


cbbPalette = [
    "#56B4E9",
    "#E69F00",
    # "#000000",
    # "#4D4D4D",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#F0E442",
]


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
            # "rolling": 3
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
#%%

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
    ylab="Normalized total daily vocal\nactivity",
    title="",
)

full_summer_plt = (
    full_summer_plt
    + geom_line(size=1)
    + scale_color_manual(
        values=["#0571b0", "#f4a582"],
        labels=["Reference", "Predictions"],
    )
    # + ggtitle("")
    + scale_x_datetime(date_breaks="4 days", labels=format_date_short)
    # + ylab("Normalized total vocal activity\nper recording")
    + theme(
        axis_title=element_text(size=12, ha="center", va="center"),
        legend_title=element_blank(),
        legend_position=(0.3, 0.3),
        legend_direction="vertical",
        figure_size=(3.5, 2.5),
        dpi=300,
        legend_text=element_text(size=12),
        plot_margin=0.05,
    )
)
#%%
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
            "period": 3,
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
# crows_df.type = crows_df.type.cat.rename_categories({"DLBD_v2": "crows"})

no_crow_row = 1
no_crows = stats["stats"].iloc[no_crow_row]
no_crows_df = stats["trends_df"][no_crow_row]["trends_df"]
no_crows_df.type = no_crows_df.type.cat.rename_categories({"ground_truth": "no_crows"})

enab_df = pd.concat(
    [crows_df, no_crows_df.loc[no_crows_df.type == "no_crows"]]
).reset_index(drop=True)


enab_df.type = enab_df.type.astype("category")
enab_df.type = enab_df.loc[:, "type"].cat.reorder_categories(
    ["ground_truth", "no_crows", "DLBD_v2"]
)


enab_plt = (
    ggplot(
        data=enab_df,
        mapping=aes(
            "date", "trend_norm", color="type", linetype="type"
        ),  # , alpha="type"),
    )
    + geom_line(size=1)  # , group="type", linetype=["solid", "dashed", "solid"])
    + scale_linetype_manual(
        name="a",
        labels=["Reference", "Reference without crows", "Predictions"],
        values=["solid", "dashed", "solid"],
    )
    + scale_color_manual(
        name="a",
        values=["#0571b0", "#0571b0", "#f4a582"],
        labels=["Reference", "Reference without crows", "Predictions"],
    )
    # + scale_alpha_manual(
    #     values=[1, 0.7, 0.7],
    #     guide=None,
    # )
    + scale_x_datetime(date_breaks="30 minutes", labels=format_date_short)
    + xlab("Date")
    + ylab("Normalized total daily vocal\nactivity")
    + theme_classic()
    + theme(
        axis_title=element_text(size=12, ha="center", va="center"),
        axis_text_x=element_text(angle=45),
        legend_title=element_blank(),
        legend_position=(0.48, 0.3),
        legend_direction="vertical",
        figure_size=(3.5, 2.5),
        dpi=300,
        legend_text=element_text(size=12),
        plot_margin=0.05,
    )
)
#%%

e_plt = pw.load_ggplot(enab_plt)
fs_plt.set_index("A)")
e_plt.set_index("B)")


final = fs_plt | e_plt

final.savefig(
    "/mnt/win/UMoncton/Doctorat/work/Articles/dlbd/figures/phenology_check.png"
)

#%%

# # * MULTIPLE THRESHOLDS

opts = {
    "evaluators": [
        {
            "type": "phenology",
            "method": "direct",
            # "activity_threshold": 0.75,
            "scenarios": {"activity_threshold": [0.5, 0.6, 0.7, 0.8, 0.9]},
        }
    ],
    "databases": [{"name": "full_summer_final"}],
}

full_summer_opts = common_utils.deep_dict_update(evaluation_config, opts, copy=True)


evaluator = SongDetectorEvaluationHandler(
    opts=full_summer_opts, dh_class=AudioDataHandler
)

stats = evaluator.evaluate()
#%%

tmp_dfs = []
gt_df = None

for nrow in range(0, stats["stats"].shape[0]):
    scenario = stats["stats"].iloc[nrow]
    # print(scenario)
    trends_df = stats["trends_df"][nrow]["trends_df"]
    trends_df.loc[:, "threshold"] = scenario.activity_threshold
    if gt_df is None:
        gt_df = trends_df.loc[trends_df.type == "ground_truth"]
    tmp_dfs.append(trends_df.loc[trends_df.type != "ground_truth"])

# tmp_dfs.append(gt_df)
all_trends = pd.concat(tmp_dfs)
all_trends.threshold = all_trends.threshold.astype("category")
print(all_trends)


#%%

thresh_plt = (
    ggplot(
        data=all_trends,
        mapping=aes("date", "trend", color="threshold"),
    )
    + geom_line(data=gt_df, color="black", size=1.5, alpha=0.5)
    + geom_line()
    # + ggtitle(plot_args.get("title", ""))
    + xlab("Date")
    + ylab("Total daily vocal activity (s)")
    # + scale_color_discrete(labels=["Model", "Reference"])
    + scale_x_datetime(labels=format_date_short)
    + scale_color_manual(values=cbbPalette, guide=None)
    + theme_classic()
    + theme(
        axis_title=element_text(size=12, ha="center", va="center"),
        axis_text_x=element_text(angle=45),
        figure_size=(3.5, 2.5),
        dpi=300,
        plot_margin=0.05,
    )
)
thresh_plt

#%%

thresh_norm_plt = (
    ggplot(
        data=all_trends,
        mapping=aes("date", "trend_norm", color="threshold"),
    )
    + geom_line(data=gt_df, color="black", size=1.5, alpha=0.5)
    + geom_line()
    # + ggtitle(plot_args.get("title", ""))
    + xlab("Date")
    + ylab("Normalized total daily vocal\nactivity")
    # + labs(linetype="Threshold")
    + scale_color_manual(values=cbbPalette, name="Threshold")
    # + scale_color_discrete(labels=["Model", "Reference"])
    + scale_x_datetime(labels=format_date_short)
    + theme_classic()
    + theme(
        axis_title=element_text(size=12, ha="center", va="center"),
        axis_text_x=element_text(angle=45),
        figure_size=(3.5, 2.5),
        dpi=300,
        legend_text=element_text(size=12),
        # plot_margin=0.05,
    )
)
thresh_norm_plt

#%%

th_plt = pw.load_ggplot(thresh_plt)
thn_plt = pw.load_ggplot(thresh_norm_plt)
th_plt.set_index("A)")
thn_plt.set_index("B)")


final = th_plt | thn_plt

final.savefig(
    "/mnt/win/UMoncton/Doctorat/work/Articles/dlbd/figures/threshold_comparison.png"
)

#%%

tags_df = evaluator.data_handler.load_dataset(
    "test",
    evaluator.data_handler.get_database("full_summer_final"),
    load_opts={"file_types": "tags_df"},
)["tags_df"]

unmatched = {}


def get_unmatched_tags(df, tags):
    tmp_unmatched = []
    tmp_tags = tags.loc[tags.recording_id == df.name]
    for tag in tmp_tags.itertuples():
        if any(
            df.loc[(df.time >= tag.tag_start) & (df.time <= tag.tag_end), "res"] == 3
        ):
            continue
        tmp_unmatched.append(tag)
    return pd.DataFrame(tmp_unmatched)


unmatched_path = Path("results/unmatched/direct_unmatched_thresh_all.feather")
if unmatched_path.exists():
    unmatched = pd.read_feather(unmatched_path)
else:
    res = []
    for nrow in range(0, stats["stats"].shape[0]):
        scenario = stats["stats"].iloc[nrow]
        matches = stats["matches"][nrow]
        matches.loc[:, "res"] = matches.events + matches.tags
        unmatched_tmp = matches.groupby("recording_id").apply(
            get_unmatched_tags, tags_df
        )
        unmatched_tmp.loc[:, "activity_threshold"] = scenario.activity_threshold
        res.append(unmatched_tmp.reset_index(drop=True))

    unmatched = pd.concat(res)
    unmatched.reset_index(drop=True).to_feather(
        file_utils.ensure_path_exists(unmatched_path, is_file=True)
    )

#%%

unmatched_stats = unmatched.groupby("activity_threshold").agg(
    {"tag_duration": ["mean", "std"]}
)
print(unmatched_stats)

unmatched.activity_threshold = unmatched.activity_threshold.astype("category")
unmatched.tag = unmatched.tag.astype("category")

# (
#     ggplot(data=unmatched, mapping=aes(x="tag", fill="activity_threshold"))
#     + geom_bar(position=position_dodge())
#     + theme(axis_text_x=element_text(angle=45))
# )
# print(scenario)
# scenario.activity_threshold

ustats = (
    unmatched.groupby(["activity_threshold", "tag"])
    .agg({"id": "count", "tag_duration": "mean"})
    .reset_index()
)

(
    ggplot(data=ustats, mapping=aes(x="tag", y="id", fill="activity_threshold"))
    + geom_bar(stat="identity", position=position_dodge())
    + theme_classic()
    + theme(
        axis_text_x=element_text(angle=45),
        figure_size=(25, 10),
    )
).save("results/umatched/tag_repartition.png")
# scenario.activity_threshold
