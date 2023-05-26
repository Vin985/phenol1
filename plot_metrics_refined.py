#%%


import ast

import pandas as pd
from dlbd.applications.phenology.phenology_evaluator import PhenologyEvaluator
from dlbd.evaluation import EVALUATORS
from mouffet import file_utils, config_utils
from plotnine import aes, geom_point, ggplot

EVALUATORS.register_evaluator(PhenologyEvaluator)


#%%

#%%

stats = pd.read_csv(
    "results/metrics/evaluation/20230214/162323_metrics_standard_refined_stats.csv"
)


if not "activity_threshold" in stats.columns:
    stats.loc[:, "activity_threshold"] = stats.evaluator_opts.apply(
        config_utils.get_option, opt_name="activity_threshold"
    )

if not "end_threshold" in stats.columns:
    stats.loc[:, "end_threshold"] = stats.evaluator_opts.apply(
        config_utils.get_option, opt_name="end_threshold"
    )

if not "min_duration" in stats.columns:
    stats.loc[:, "min_duration"] = stats.evaluator_opts.apply(
        config_utils.get_option, opt_name="min_duration"
    )

#%%

plt = (
    ggplot(
        stats, mapping=aes(x="activity_threshold", y="f1_score", color="end_threshold")
    )
    + geom_point()
    # + geom_point(mapping=aes(y="eucl_distance"), color="red")
)
plt.save(
    file_utils.ensure_path_exists(
        "results/plots/metrics/f1_score_standard_metrics_refined_plot.png", is_file=True
    ),
    width=10,
    height=10,
)


plt = (
    ggplot(
        stats,
        mapping=aes(x="activity_threshold", y="eucl_distance", color="end_threshold"),
    )
    + geom_point()
    # + geom_point(mapping=aes(y="eucl_distance"), color="red")
)
plt.save(
    file_utils.ensure_path_exists(
        "results/plots/metrics/eucl_distance_score_standard_metrics_refined_plot.png",
        is_file=True,
    ),
    width=10,
    height=10,
)

plt = (
    ggplot(
        stats,
        mapping=aes(
            x="activity_threshold", y="eucl_distance_norm", color="end_threshold"
        ),
    )
    + geom_point()
    # + geom_point(mapping=aes(y="eucl_distance"), color="red")
)
plt.save(
    file_utils.ensure_path_exists(
        "results/plots/metrics/eucl_distance_norm_standard_metrics_refined_plot.png",
        is_file=True,
    ),
    width=10,
    height=10,
)

plt = (
    ggplot(
        stats,
        mapping=aes(x="activity_threshold", y="IoU", color="end_threshold"),
    )
    + geom_point()
    # + geom_point(mapping=aes(y="eucl_distance"), color="red")
)
plt.save(
    file_utils.ensure_path_exists(
        "results/plots/metrics/IoU_standard_metrics_refined_plot.png",
        is_file=True,
    ),
    width=10,
    height=10,
)

#%%
