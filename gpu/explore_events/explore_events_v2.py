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

res_dir = Path(".")
events_dir = res_dir / "events"

# site_data = pd.read_csv(
#     "/mnt/win/UMoncton/Doctorat/data/acoustic/deployment data/sites_deployment_all.csv"
# )


get_audio = True

hd_root = "/mnt/Backup/PhD/Acoustics/"
dest_dir = res_dir / "events_to_explore"

# dest_event_file = file_utils.ensure_path_exists(
#     dest_dir / "events_to_explore.csv", is_file=True
# )

dest_event_file = Path("events_to_explore.feather")

nfiles = 20
overwrite = False
min_recordings_required = 960  # min 20 days of recording
# seed = 1285467
seed = 795243
overwrite = False

opts = {
    "method": "standard",
    # "method": "direct",
    "activity_threshold": 0.9,
    "min_duration": 0.4,
    "end_threshold": 0.5,
}

method_id = f"{opts['method']}_{opts['activity_threshold']}"
if opts["method"] == "standard":
    method_id += f"_{opts['end_threshold']}"

TOTAL_DURATION = 600
PADDING_DURATION = 0.5
N_FILES = 300


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
    # dates = {
    #     cat: datetime.fromtimestamp(int(Path(cat).stem, 16))
    #     for cat in df.recording_id.cat.categories
    # }
    # df["full_date"] = None
    # for k, v in dates.items():
    #     df.loc[df.recording_id == k, "full_date"] = v

    df["full_date"] = df.recording_id.path.stem.apply(
        lambda x: datetime.fromtimestamp(int(x, 16))
    )

    df["date"] = pd.to_datetime(df["full_date"].dt.strftime("%Y%m%d"))
    df["date_hour"] = pd.to_datetime(
        df["full_date"].dt.strftime("%Y%m%d_%H"), format="%Y%m%d_%H"
    )
    return df


#%%

if dest_event_file.exists():

    mixed = pd.read_feather(dest_event_file)

else:

    event_files = events_dir.glob(f"*_{method_id}_*.feather")

    site_data["depl_start"] = pd.to_datetime(site_data["depl_start"], format="%d-%m-%Y")
    site_data["depl_end"] = pd.to_datetime(site_data["depl_end"], format="%d-%m-%Y")

    current_module = sys.modules[__name__]
    res = []
    all_events = []
    for event_file in event_files:
        print(f"Processing {event_file}")
        df = pd.read_feather(event_file)
        if len(df.recording_id.unique()) < min_recordings_required:
            common_utils.print_warning("Not enough recordings found, skipping")
            continue

        splits = df.iloc[0].recording_id.split("/")
        year = int(splits[-4])
        site = splits[-3]
        plot = splits[-2]

        tmp_year = 2019 if year >= 2019 else year
        func_name = "extract_recording_info_" + str(tmp_year)
        df = getattr(current_module, func_name)(df)

        site_info = site_data.loc[
            (site_data.year == year) & (site_data["plot"] == plot),
        ].iloc[0]
        if not pd.isnull(site_info["depl_start"]):
            df = df.loc[df.date_hour >= site_info["depl_start"]]
        if not pd.isnull(site_info["depl_end"]):
            df = df.loc[df.date_hour <= site_info["depl_end"]]

        df["year"] = year
        df["site"] = site
        df["plot"] = plot

        all_events.append(df)

    all_events_df = pd.concat(all_events)

    mixed = all_events_df.sample(n=min(N_FILES, df.shape[0]), random_state=seed)
    mixed.reset_index().to_feather(dest_event_file, index=False)


if get_audio:
    # mixed.recording_id = mixed.recording_id.str.replace("/mnt", "/mnt/vin/Backup")

    extract_nb = 1
    tmp_res = []
    for event in mixed.itertuples():
        print(
            f"{event.year}, {event.plot}: Processing extract in recording {event.recording_id}"
        )
        tmp = event._asdict()

        # file_name = dest_path.stem
        # if event.year == "2018":
        #     dt = datetime.fromtimestamp(int(file_name, 16))
        #     file_name = dt.strftime("%Y%m%d_%H%M%S")
        #     tmp["date"] = file_name

        file_name = f"{event.year}_{event.plot}_{event.date_hour.strftime('%Y%m%d_%H%M%S')}_{extract_nb}"

        dest_path = file_utils.ensure_path_exists(
            dest_dir / (file_name + dest_path.suffix), is_file=True
        )
        if not dest_path.exists() or overwrite:
            audio = Audio(event.recording_id)
            audio.write(
                dest_path,
                start=max(0, event.event_start - PADDING_DURATION),
                end=min(300, event.event_end + PADDING_DURATION),
                seconds=True,
            )
        else:
            print("Already exists skipping!")
        # total_dur += event.event_duration
        # if total_dur >= TOTAL_DURATION:
        #     break
        extract_nb += 1
        tmp["dest_path"] = dest_path
        tmp_res.append(tmp)
    tmp_df = pd.DataFrame(tmp_res)
    res.append(tmp_df.reset_index(drop=True))

    res_df = pd.concat(res)
    res_df.reset_index(drop=True).to_csv(dest_dir + "/files.csv")
# tmp = (
#     df.groupby("recording_id")
#     .apply(file_event_duration)
#     .reset_index()
#     .rename(columns={0: "duration"})
#     .sort_values("duration", ascending=False)
#     .iloc[0:(nfiles)]
# )

# if not tmp[tmp.duration > 300].empty:
#     print("event_file")

# splits = tmp.recording_id.str.split("/")

# tmp["year"] = splits.str[-4]
# tmp["site"] = splits.str[-3]
# tmp["plot"] = splits.str[-2]
# # tmp["name"] = splits.str[-1].str.split(".").str[0]

# year = tmp.year.iloc[0]
# if int(year) > 2019:
#     year = 2019
# func_name = "extract_recording_info_" + str(year)
# tmp = getattr(current_module, func_name)(tmp)

# %%
