#%%


import pandas as pd
from dlbd.applications.phenology.phenology_evaluator import PhenologyEvaluator
from dlbd.evaluation import EVALUATORS
from mouffet import config_utils, file_utils
from plotnine import aes, facet_wrap, geom_point, ggplot

EVALUATORS.register_evaluator(PhenologyEvaluator)


#%%

#%%

# stats = pd.read_csv("results/metrics/evaluation/gpu_metrics_standard_stats.csv")

stats = pd.read_csv(
    "/mnt/win/UMoncton/Doctorat/dev/phenol1/results/metrics_v2/evaluation/20230417/134726_metrics_standard_v2_stats.csv"
)

dest_dir = "results/plots/metrics_v2/global/"


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
    + facet_wrap("~min_duration")
    # + geom_point(mapping=aes(y="eucl_distance"), color="red")
)
print(plt)
plt.save(
    file_utils.ensure_path_exists(
        dest_dir + "f1_score_standard_metrics_plot.png", is_file=True
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
    + facet_wrap("~min_duration")
    # + geom_point(mapping=aes(y="eucl_distance"), color="red")
)
print(plt)
plt.save(
    file_utils.ensure_path_exists(
        dest_dir + "eucl_distance_score_standard_metrics_plot.png",
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
    + facet_wrap("~min_duration")
    # + geom_point(mapping=aes(y="eucl_distance"), color="red")
)
print(plt)
plt.save(
    file_utils.ensure_path_exists(
        dest_dir + "eucl_distance_norm_standard_metrics_plot.png",
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
    + facet_wrap("~min_duration")
    # + geom_point(mapping=aes(y="eucl_distance"), color="red")
)
print(plt)
plt.save(
    file_utils.ensure_path_exists(
        dest_dir + "IoU_standard_metrics_plot.png",
        is_file=True,
    ),
    width=10,
    height=10,
)

#%%

plt = (
    ggplot(stats, mapping=aes(x="eucl_distance_norm"))
    + geom_point(mapping=aes(y="f1_score"), color="blue")
    # + geom_point(mapping=aes(y="eucl_distance"), color="red")
)
print(plt)

plt = (
    ggplot(stats, mapping=aes(x="eucl_distance"))
    + geom_point(mapping=aes(y="f1_score"), color="red")
    # + geom_point(mapping=aes(y="eucl_distance"), color="red")
)
print(plt)

plt = (
    ggplot(stats, mapping=aes(x="precision"))
    + geom_point(mapping=aes(y="f1_score"), color="red")
    # + geom_point(mapping=aes(y="eucl_distance"), color="red")
)
print(plt)

plt = (
    ggplot(stats, mapping=aes(x="IoU"))
    + geom_point(mapping=aes(y="f1_score"), color="red")
    # + geom_point(mapping=aes(y="eucl_distance"), color="red")
)
print(plt)

plt = (
    ggplot(stats, mapping=aes(x="activity_threshold"))
    + geom_point(mapping=aes(y="f1_score"), color="blue")
    # + geom_point(mapping=aes(y="eucl_distance_norm"), color="green")
    + geom_point(mapping=aes(y="precision"), color="black")
    + geom_point(mapping=aes(y="IoU"), color="orange")
    + geom_point(mapping=aes(y="eucl_distance"), color="red")
)
print(plt)
