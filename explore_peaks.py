import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import shutil
from mouffet.utils import common_utils, file_utils
from pandas_path import path

res_dir = Path("/mnt/win/UMoncton/Doctorat/dev/phenol1/results")
events_dir = res_dir / "events"
hd_root = "/media/vin/Backup/PhD/Acoustics/"
dest_dir = "/media/vin/Backup/peaks/"
nfiles = 20
overwrite = False
min_recordings_required = 960  # min 20 days of recording

event_files = events_dir.glob("*.feather")


def file_event_duration(df):
    if df.shape[0] > 0:
        step = df.time.iloc[1] - df.time.iloc[0]
        return df.events.sum() / 2 * step
    return 0


def extract_recording_info_2019(df):
    df[["date", "time"]] = df.recording_id.path.stem.str.split("_", expand=True)
    df = df.assign(full_date=[str(x) + "_" + y for x, y in zip(df["date"], df["time"])])
    df["full_date"] = pd.to_datetime(df["full_date"], format="%Y%m%d_%H%M%S")
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df["date_hour"] = pd.to_datetime(
        df["full_date"].dt.strftime("%Y%m%d_%H"), format="%Y%m%d_%H"
    )
    df.drop(columns=["time"])
    return df


def extract_recording_info_2018(df):
    df["full_date"] = df.recording_id.path.stem.apply(
        lambda x: datetime.fromtimestamp(int(x, 16))
    )

    df["date"] = pd.to_datetime(df["full_date"].dt.strftime("%Y%m%d"))
    df["date_hour"] = pd.to_datetime(
        df["full_date"].dt.strftime("%Y%m%d_%H"), format="%Y%m%d_%H"
    )
    return df


current_module = sys.modules[__name__]
res = []
for event_file in event_files:
    print(f"Processing {event_file}")
    df = pd.read_feather(event_file)
    if len(df.recording_id.unique()) < min_recordings_required:
        common_utils.print_warning("Not enough recordings found, skipping")
        continue
    df.recording_id = df.recording_id.str.replace("/mnt", "/media/vin/Backup")
    tmp = (
        df.groupby("recording_id")
        .apply(file_event_duration)
        .reset_index()
        .rename(columns={0: "duration"})
        .sort_values("duration", ascending=False)
        .iloc[0:(nfiles)]
    )

    if not tmp[tmp.duration > 300].empty:
        print("event_file")

    splits = tmp.recording_id.str.split("/")

    tmp["year"] = splits.str[-4]
    tmp["site"] = splits.str[-3]
    tmp["plot"] = splits.str[-2]
    # tmp["name"] = splits.str[-1].str.split(".").str[0]

    year = tmp.year.iloc[0]
    if int(year) > 2019:
        year = 2019
    func_name = "extract_recording_info_" + str(year)
    tmp = getattr(current_module, func_name)(tmp)

#     res.append(tmp)
#     for row in tmp.itertuples():
#         dest_path = file_utils.ensure_path_exists(
#             Path(row.recording_id.replace(hd_root, dest_dir)), is_file=True
#         )
#         if not dest_path.exists() or overwrite:
#             print(f"Copying file {dest_path}")
#             shutil.copy(row.recording_id, dest_path)
# res_df = pd.concat(res)
# res_df.to_csv(dest_dir + "peaks.csv", index=False)
# print(res_df)
