#%%
import ast

import mouffet.utils.file as file_utils
import pandas as pd
from plotnine import *


#%%

# stats = pd.read_csv("results/metrics/evaluation/gpu_metrics_standard_stats.csv")

# stats =


method = "direct_positive"

stats_paths = {
    "standard": "/mnt/win/UMoncton/Doctorat/dev/phenol1/final/results/metrics/evaluation/20230502/144515_metrics_standard_final_stats.csv",
    "direct": "/mnt/win/UMoncton/Doctorat/dev/phenol1/final/results/metrics/evaluation/20230511/151955_metrics_direct_final_stats.csv",
    "direct_positive": "/mnt/win/UMoncton/Doctorat/dev/phenol1/final/results/metrics/evaluation/20230511/152731_metrics_direct_positive_final_stats.csv",
}

stats = pd.read_csv(stats_paths[method])

dest_dir = f"results/plots/metrics/{method}/"


def extract_option(x, opt_name):
    return ast.literal_eval(x).get(opt_name, "")


if not "activity_threshold" in stats.columns:
    stats.loc[:, "activity_threshold"] = stats.evaluator_opts.apply(
        extract_option, opt_name="activity_threshold"
    )

if not "end_threshold" in stats.columns:
    stats.loc[:, "end_threshold"] = stats.evaluator_opts.apply(
        extract_option, opt_name="end_threshold"
    )

if not "min_duration" in stats.columns:
    stats.loc[:, "min_duration"] = stats.evaluator_opts.apply(
        extract_option, opt_name="min_duration"
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

plt = (
    ggplot(
        stats, mapping=aes(x="activity_threshold", y="f1_score", color="end_threshold")
    )
    + geom_point()
    + facet_wrap("~min_duration")
    # + geom_point(mapping=aes(y="eucl_distance"), color="red")
)
plt.save(
    file_utils.ensure_path_exists(dest_dir + "f1_score_metrics_plot.png", is_file=True),
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
plt.save(
    file_utils.ensure_path_exists(
        dest_dir + "eucl_distance_score_metrics_plot.png",
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
plt.save(
    file_utils.ensure_path_exists(
        dest_dir + "eucl_distance_norm_metrics_plot.png",
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
plt.save(
    file_utils.ensure_path_exists(
        dest_dir + "IoU_metrics_plot.png",
        is_file=True,
    ),
    width=10,
    height=10,
)

#%%

plt = (
    ggplot(stats, mapping=aes(x="f1_score"))
    + geom_point(mapping=aes(y="eucl_distance_norm"), color="blue")
    # + geom_point(mapping=aes(y="eucl_distance"), color="red")
)
plt.save(
    file_utils.ensure_path_exists(
        dest_dir + "F1_distance_norm_metrics_plot.png",
        is_file=True,
    ),
    width=10,
    height=10,
)

plt = (
    ggplot(stats, mapping=aes(x="f1_score"))
    + geom_point(mapping=aes(y="eucl_distance"), color="red")
    # + geom_point(mapping=aes(y="eucl_distance"), color="red")
)
plt.save(
    file_utils.ensure_path_exists(
        dest_dir + "F1_distance_metrics_plot.png",
        is_file=True,
    ),
    width=10,
    height=10,
)

# plt = (
#     ggplot(stats, mapping=aes(x="precision"))
#     + geom_point(mapping=aes(y="f1_score"), color="red")
#     # + geom_point(mapping=aes(y="eucl_distance"), color="red")
# )
# plt.save(
#     file_utils.ensure_path_exists(
#         dest_dir + "precision_F1_metrics_plot.png",
#         is_file=True,
#     ),
#     width=10,
#     height=10,
# )


plt = (
    ggplot(stats, mapping=aes(x="IoU"))
    + geom_point(mapping=aes(y="f1_score"), color="red")
    # + geom_point(mapping=aes(y="eucl_distance"), color="red")
)
plt.save(
    file_utils.ensure_path_exists(
        dest_dir + "Iou_F1_metrics_plot.png",
        is_file=True,
    ),
    width=10,
    height=10,
)


# plt = (
#     ggplot(stats, mapping=aes(x="activity_threshold"))
#     + geom_point(mapping=aes(y="f1_score"), color="blue")
#     # + geom_point(mapping=aes(y="eucl_distance_norm"), color="green")
#     + geom_point(mapping=aes(y="precision"), color="black")
#     + geom_point(mapping=aes(y="IoU"), color="orange")
#     + geom_point(mapping=aes(y="eucl_distance"), color="red")
# )

# plt.save(
#     file_utils.ensure_path_exists(
#         dest_dir + "all_metrics_plot.png",
#         is_file=True,
#     ),
#     width=10,
#     height=10,
# )
