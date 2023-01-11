import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from dlbd.applications.phenology import PhenologyEvaluator
from dlbd.evaluation import EVALUATORS
from dlbd.utils.plot_utils import format_date_short
from mouffet.utils import common_utils, file_utils
from pandas_path import path  # pylint: disable=unused-import
from plotnine import *

from utils import format_date_short, read_trends

EVALUATORS.register_evaluator(PhenologyEvaluator)
preds_root = Path(
    "/mnt/win/UMoncton/OneDrive - Université de Moncton/Data/Results/predictions"
)

site = "IGLO"
plot = "B"
plot_name = "_".join([site, plot])
year = 2018


file_name = f"{year}_{plot_name}_predictions_overlap-0.5.feather"


text_size = 16
title_size = 18
legend_title_margin = {"b": 15}
activity_threshold = 0.7
method = "citynet"
trend = "trend"


site_data = pd.read_csv(
    "/mnt/win/UMoncton/Doctorat/data/acoustic/deployment data/sites_deployment_all.csv"
)
site_data["depl_start"] = pd.to_datetime(site_data["depl_start"], format="%d-%m-%Y")
site_data["depl_end"] = pd.to_datetime(site_data["depl_end"], format="%d-%m-%Y")


res_dir = Path("/mnt/win/UMoncton/Doctorat/dev/phenol1/results")
events_dir = res_dir / "events"
plot_dir = res_dir / "plots"


opts = {
    "method": "standard",
    # "method": "standard",
    "activity_threshold": 0.9,
    "min_duration": 0.1,
    "end_threshold": 0.35,
    "daily_aggregation": "sum",
    "recording_info_type": "audiomoth2019",
}

thresholds = common_utils.range_list(0.5, 0.99, 0.1)

errors = []

min_recordings_required = 480  # Get minimum 10 days worth of recordings


def create_error(year, site, plot, error, file_name=""):
    msg = f"{error} for {plot} in {year}"
    common_utils.print_warning(msg)
    return {"year": year, "site": site, "plot": plot, "error": error, "file": file_name}


#%%

print(f"Processing file {str(file_name)}")
site_info = site_data.loc[
    (site_data.year == year) & (site_data["plot"] == plot_name),
]
if site_info.empty:
    errors.append(
        create_error(year, site, plot_name, "Recordings not found in deployment data")
    )
else:
    site_info = site_info.iloc[0].to_dict()

int_year = int(year)
if int_year > 2019:
    int_year = 2019
opts["recording_info_type"] = f"audiomoth{int_year}"
if not pd.isnull(site_info["depl_start"]):
    opts["depl_start"] = site_info["depl_start"]
else:
    errors.append(create_error(year, site, plot_name, "No deployment start provided"))
if not pd.isnull(site_info["depl_end"]):
    opts["depl_end"] = site_info["depl_end"]
else:
    errors.append(create_error(year, site, plot_name, "No deployment end provided"))
opts["plot"] = plot_name

events_file = (
    events_dir
    / f"{year}_{site}-{plot}_{opts['method']}_{opts['activity_threshold']}_events.feather"
)

preds = pd.read_feather(preds_root / file_name)
preds = preds.rename(columns={"recording_path": "recording_id"})

res = []
for t in thresholds:
    print(f"Computing events for threshold {t}")
    tmp_opts = opts.copy()
    opts["activity_threshold"] = t
    events = EVALUATORS[opts["method"]].filter_predictions(preds, opts)
    daily_activity = EVALUATORS["phenology"].get_daily_trends(events, opts, "event")
    df = daily_activity["trends_df"]
    df["threshold"] = t
    res.append(df)

res_df = pd.concat(res)

#%%


res_df = res_df.astype({"threshold": "category"})
gb = res_df.groupby("date")
df_mean = gb.apply("mean").reset_index()
df_sd = gb.apply("std").reset_index()
df_sd["trend_min"] = df_mean["trend"] - df_sd["trend"]
df_sd["trend_max"] = df_mean["trend"] + df_sd["trend"]
plt = (
    ggplot()
    + geom_line(data=res_df, mapping=aes("date", "trend", colour="threshold"))
    + geom_line(data=df_mean, mapping=aes("date", "trend"), colour="#000000", size=2)
    + geom_ribbon(
        data=df_sd,
        mapping=aes(x="date", ymin="trend_min", ymax="trend_max"),
        fill="#DDDDDD",
        alpha=0.5,
    )
    + ggtitle(f"Daily mean acoustic activity per recording for {site}_{plot} in {year}")
    + xlab("Date")
    + ylab("Daily mean activity per recording (s)")
    + scale_x_datetime(labels=format_date_short)
    + theme_classic()
    + theme(axis_text_x=element_text(angle=45))
).save(
    plot_dir / f"{plot_name}_{year}_various_thresholds_sum2.png",
    width=12,
    height=7,
)

plt
