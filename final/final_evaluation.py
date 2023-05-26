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

models_dir = "/mnt/win/UMoncton/Doctorat/dev/phenol1/resources/models"

evaluation_config_path = "config/evaluation_config.yaml"

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
