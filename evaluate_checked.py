import mouffet.utils.file as file_utils
import pandas as pd
from dlbd.data.audio_data_handler import AudioDataHandler
from dlbd.data.tag_utils import flatten_tags
from dlbd.evaluation import EVALUATORS
from dlbd.evaluation.song_detector_evaluation_handler import (
    SongDetectorEvaluationHandler,
)
from mouffet import common_utils, config_utils
from mouffet.evaluation import Evaluator


class PresenceEvaluator(Evaluator):

    NAME = "presence"

    def requires(self, options):
        return EVALUATORS[options["method"]].requires(options) + ["metadata"]

    def file_event_duration(self, df, method):
        return EVALUATORS[method].file_event_duration(df)

    def file_tag_duration(self, df, method=None):
        flattened = flatten_tags(df)
        return flattened.tag_duration.sum()

    def get_stats(self, match_df, pred_col, gt_col="has_bird"):
        res = match_df[gt_col] + match_df[pred_col]

        n_true_positives = len(res[res == 3])
        n_true_negatives = len(res[res == 0])
        n_false_positives = len(res[res == 2])
        n_false_negatives = len(res[res == 1])

        precision = round(n_true_positives / (n_true_positives + n_false_positives), 3)
        recall = round(n_true_positives / (n_true_positives + n_false_negatives), 3)
        f1_score = round(2 * precision * recall / (precision + recall), 3)

        stats = {
            "n_true_positives": n_true_positives,
            "n_false_positives": n_false_positives,
            "n_true_negatives": n_true_negatives,
            "n_false_negatives": n_false_negatives,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }

        common_utils.print_warning(
            # f"Threshold: {threshold},"
            f'Precision: { stats["precision"]};'
            + f' Recall: {stats["recall"]};'
            + f' F1_score: {stats["f1_score"]}'
        )
        return pd.DataFrame([stats])  # , predictions

    def evaluate(self, data, options, infos):
        if not self.check_database(data, options, infos):
            return {}
        predictions, gt = data
        metadata = gt["metadata"]
        method = options["method"]
        events = EVALUATORS[method].filter_predictions(predictions, options)

        # matches = stats["matches"]
        tags = gt["tags_df"]

        files_with_birds = tags["recording_id"].unique().tolist()
        files_detected = events["recording_id"].unique().tolist()

        tmp = []
        for f in metadata:
            tmp.append(
                {
                    "recording_id": f["file_path"],
                    "has_bird": int(f["file_path"] in files_with_birds),
                    "previous_event": 2,
                    "new_event": int(f["file_path"] in files_detected) * 2,
                }
            )

        match_df = pd.DataFrame(tmp)

        old_stats = self.get_stats(match_df, "previous_event")
        new_stats = self.get_stats(match_df, "new_event")

        return {"stats": pd.concat([old_stats, new_stats])}


EVALUATORS.register_evaluator(PresenceEvaluator)


models_dir = "resources/models"
evaluation_config_path = "config/presence/evaluation_config.yaml"

# models_dir = "/home/vin/Desktop/results/22_checked_all/models"
# evaluation_config_path = "config/test_presence/evaluation_config.yaml"

evaluation_config = file_utils.load_config(evaluation_config_path)


evaluation_config["models_list_dir"] = models_dir

evaluation_config = config_utils.get_models_conf(
    evaluation_config,
)


evaluator = SongDetectorEvaluationHandler(
    opts=evaluation_config, dh_class=AudioDataHandler
)

stats = evaluator.evaluate()
