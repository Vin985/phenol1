#################################################################################################
## This file saves audio extracts that were classified as events by our model for verification ##
#  A total of at least 10 minutes during the deployment period per plot is saved               ##
# Plots with less than 20 days of recording are not considered                                 ##
#################################################################################################

#%%
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from mouffet.utils import common_utils, file_utils
from pandas_path import path
from pysoundplayer.audio import Audio

PADDING_DURATION = 0.5

audio_src = Path("/mnt/win/UMoncton/Doctorat/data/dl_training/raw/predictions_raw")
events_df_path = Path(
    "/mnt/win/UMoncton/Doctorat/dev/phenol1/results/v2/events_to_explore/events_to_explore.csv"
)

raw_dir = audio_src / "events_files"
dest_dir = audio_src / "extracts"

events_df = pd.read_csv(events_df_path)
events_df.loc[:, "file_name"] = events_df.recording_id.path.name
events_df.drop_duplicates("file_name", keep=False).reset_index(drop=True)


events_df.date_hour = pd.to_datetime(events_df.date_hour, format="%Y-%m-%d %H:%M:%S")

events_df.loc[:, "new_file_path"] = (
    str(raw_dir)
    + "/"
    + events_df.year.astype(str)
    + "_"
    + events_df["plot"].astype(str)
    + "_"
    + events_df.date_hour.dt.strftime("%Y%m%d_%H%M%S")
    + ".WAV"
)

events_df.to_feather(audio_src / "events_df.feather")

#%%

extract_nb = 1
tmp_res = []
for event in events_df.itertuples():
    print(
        f"{event.year}, {event.plot}: Processing extract in recording {event.recording_id}"
    )
    tmp = event._asdict()

    file_name = f"{event.year}_{event.plot}_{event.date_hour.strftime('%Y%m%d_%H%M%S')}_{extract_nb}.WAV"

    dest_path = file_utils.ensure_path_exists(dest_dir / file_name, is_file=True)
    if not dest_path.exists():
        audio = Audio(raw_dir / event.file_name)
        audio.write(
            dest_path,
            start=max(0, event.event_start - PADDING_DURATION),
            end=min(300, event.event_end + PADDING_DURATION),
            seconds=True,
        )
    else:
        print("Already exists skipping!")
    extract_nb += 1
    tmp["dest_path"] = dest_path
    tmp_res.append(tmp)

# %%
