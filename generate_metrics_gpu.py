#%%

from dlbd.applications.phenology.phenology_evaluator import PhenologyEvaluator
from dlbd.data.audio_data_handler import AudioDataHandler
from dlbd.evaluation import EVALUATORS
from dlbd.evaluation.song_detector_evaluation_handler import (
    SongDetectorEvaluationHandler,
)
from mouffet import config_utils, file_utils

EVALUATORS.register_evaluator(PhenologyEvaluator)


#%%

models_dir = "resources/models"

evaluation_config_path = "config/metrics_gpu/evaluation_config.yaml"

evaluation_config = file_utils.load_config(evaluation_config_path)

print(evaluation_config)

evaluation_config = config_utils.get_models_conf(
    evaluation_config,
)


evaluator = SongDetectorEvaluationHandler(
    opts=evaluation_config, dh_class=AudioDataHandler
)

stats = evaluator.evaluate()


#%%

# stats = pd.read_csv("results/metrics/evaluation/20230210/144748_metrics_stats.csv")

# plt = (
#     ggplot(stats, mapping=aes(x="eucl_distance_norm"))
#     + geom_point(mapping=aes(y="f1_score"), color="blue")
#     # + geom_point(mapping=aes(y="eucl_distance"), color="red")
# )
# print(plt)

# plt = (
#     ggplot(stats, mapping=aes(x="eucl_distance"))
#     + geom_point(mapping=aes(y="f1_score"), color="red")
#     # + geom_point(mapping=aes(y="eucl_distance"), color="red")
# )
# print(plt)

# plt = (
#     ggplot(stats, mapping=aes(x="precision"))
#     + geom_point(mapping=aes(y="f1_score"), color="red")
#     # + geom_point(mapping=aes(y="eucl_distance"), color="red")
# )
# print(plt)

# plt = (
#     ggplot(stats, mapping=aes(x="IoU"))
#     + geom_point(mapping=aes(y="f1_score"), color="red")
#     # + geom_point(mapping=aes(y="eucl_distance"), color="red")
# )
# print(plt)

# plt = (
#     ggplot(stats, mapping=aes(x="activity_threshold"))
#     + geom_point(mapping=aes(y="f1_score"), color="blue")
#     # + geom_point(mapping=aes(y="eucl_distance_norm"), color="green")
#     + geom_point(mapping=aes(y="precision"), color="black")
#     + geom_point(mapping=aes(y="IoU"), color="orange")
#     + geom_point(mapping=aes(y="eucl_distance"), color="red")
# )
# print(plt)
