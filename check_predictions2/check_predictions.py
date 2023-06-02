#%%
import pandas as pd

from plotnine import *
from pathlib import Path
from pandas_path import path

original_events = pd.read_feather(
    "/mnt/win/UMoncton/Doctorat/data/dl_training/raw/predictions_raw/events_df.feather"
)
raw_predictions = pd.read_feather(
    "results/predictions/raw_predictions_overlap-0.75.feather"
)
extract_predictions = pd.read_feather(
    "results/predictions/extracts_predictions_overlap-0.75.feather"
)

old_preds_dir = Path("/home/vin/Desktop/results/predictions_v2")

new_gpu_predictions = pd.read_feather(
    "/home/vin/Desktop/results/new_check_gpu_predictions/raw_predictions_overlap-0.75.feather"
)

original_events.loc[:, "new_file_name"] = original_events.new_file_path.path.stem
raw_predictions.loc[:, "file_name"] = raw_predictions.recording_path.path.name
extract_predictions.loc[:, "file_name"] = extract_predictions.recording_path.path.stem
new_gpu_predictions.loc[:, "file_name"] = new_gpu_predictions.recording_path.path.name


#%%

for event in original_events.itertuples():
    print(event.file_name)

    # * Old gpu predictions
    old_preds_df = pd.read_feather(
        old_preds_dir / f"{event.year}_{event.plot}_predictions_overlap-0.75.feather"
    )
    old_preds = (
        old_preds_df.loc[
            (old_preds_df.recording_path.str.endswith(event.file_name))
            & (old_preds_df.time >= event.event_start - 0.5)
            & (old_preds_df.time <= event.event_end + 0.5),
        ]
        .copy()
        .reset_index(drop=True)
    )
    old_preds.loc[:, "type"] = "old"

    # * Predictions for the whole file, laptop
    raw = raw_predictions.loc[raw_predictions.file_name == event.file_name].copy()
    raw_preds = raw.loc[
        (raw.time >= event.event_start - 0.5) & (raw.time <= event.event_end + 0.5),
    ].copy()
    raw_preds.loc[:, "type"] = "raw"

    # * Predictions for the single extract
    extract = extract_predictions.loc[
        extract_predictions.file_name.str.startswith(event.new_file_name)
    ].copy()
    extract.loc[:, "time"] += event.event_start - 0.5
    extract.loc[:, "type"] = "extract"

    # * New predictions from gpu machine
    new = new_gpu_predictions.loc[
        new_gpu_predictions.file_name == event.file_name
    ].copy()
    new_preds = new.loc[
        (new.time >= event.event_start - 0.5) & (new.time <= event.event_end + 0.5),
    ].copy()
    new_preds.loc[:, "type"] = "new"

    plt_df = pd.concat([old_preds, extract, new_preds])

    plt = (
        ggplot(data=plt_df, mapping=aes(x="time", y="activity", color="type"))
        + geom_path()
    )

    print(plt)

    break
