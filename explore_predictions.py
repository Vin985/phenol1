#%%
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

EVALUATORS.register_evaluator(PhenologyEvaluator)
preds_root = Path(
    "/mnt/win/UMoncton/OneDrive - Université de Moncton/Data/Results/predictions"
)


site_data = pd.read_csv(
    "/mnt/win/UMoncton/Doctorat/data/acoustic/deployment data/sites_deployment_all.csv"
)
site_data["depl_start"] = pd.to_datetime(site_data["depl_start"], format="%d-%m-%Y")
site_data["depl_end"] = pd.to_datetime(site_data["depl_end"], format="%d-%m-%Y")


res_dir = Path("/mnt/win/UMoncton/Doctorat/dev/phenol1/results")
events_dir = res_dir / "events"

pred_files = preds_root.glob("*.feather")

default_opts = {
    "event_threshold": 0.5,
    "method": "citynet",
    # "method": "standard",
    "activity_threshold": 0.9,
    "min_duration": 0.1,
    "end_threshold": 0.3,
    "recording_info_type": "audiomoth2019",
}

errors = []

min_recordings_required = 480  # Get minimum 10 days worth of recordings


def create_error(year, site, plot, error, file_name=""):
    msg = f"{error} for {plot} in {year}"
    common_utils.print_warning(msg)
    return {"year": year, "site": site, "plot": plot, "error": error, "file": file_name}


#%%

plts = []
agg_plts = {}
agg_events = {}
for pred_file in pred_files:
    # if len(errors) > 2:
    #     break
    print(f"Processing file {str(pred_file)}")
    infos = pred_file.stem.split("_")

    year = int(infos[0])
    site = infos[1]
    plot = infos[2]
    plot_name = site + "_" + plot
    site_info = site_data.loc[
        (site_data.year == year) & (site_data["plot"] == plot_name),
    ]
    if site_info.empty:
        errors.append(
            create_error(
                year, site, plot_name, "Recordings not found in deployment data"
            )
        )
        continue
    else:
        site_info = site_info.iloc[0].to_dict()

    opts = default_opts.copy()
    int_year = int(year)
    if int_year > 2019:
        int_year = 2019
    opts["recording_info_type"] = f"audiomoth{int_year}"
    if not pd.isnull(site_info["depl_start"]):
        opts["depl_start"] = site_info["depl_start"]
    else:
        errors.append(
            create_error(year, site, plot_name, "No deployment start provided")
        )
    if not pd.isnull(site_info["depl_end"]):
        opts["depl_end"] = site_info["depl_end"]
    else:
        errors.append(create_error(year, site, plot_name, "No deployment end provided"))
    opts["plot"] = plot_name

    events_file = (
        events_dir
        / f"{year}_{site}-{plot}_{opts['method']}_{opts['activity_threshold']}_events.feather"
    )

    if events_file.exists():
        events = pd.read_feather(events_file)
    else:
        preds = pd.read_feather(pred_file)
        preds = preds.rename(columns={"recording_path": "recording_id"})
        try:
            events = EVALUATORS[opts["method"]].filter_predictions(preds, opts)
        except IndexError:
            errors.append(
                create_error(
                    year, site, plot_name, "Problem encountered when applying evaluator"
                )
            )
            continue
        events.to_feather(file_utils.ensure_path_exists(events_file, is_file=True))

    if not events.empty:
        if len(events.recording_id.unique()) < min_recordings_required:
            errors.append(
                create_error(year, site, plot_name, "Not enough recordings found")
            )
            continue

        try:
            daily_activity = EVALUATORS["phenology"].get_daily_trends(
                events, opts, "event"
            )
        except ValueError:
            errors.append(
                create_error(year, site, plot_name, "Error when getting daily trends")
            )
            continue

        df = daily_activity["trends_df"]

        #' Check if data is saved in the correct year
        if df["date"].dt.year[0] < int(year):
            errors.append(create_error(year, site, plot_name, "Wrong year"))
            continue

        if year not in agg_plts:
            agg_plts[year] = []
        df["site"] = site
        agg_plts[year].append(df)

        plt = (
            ggplot(
                data=df,
                mapping=aes("date", "trend"),
            )
            + geom_line()
            + ggtitle(
                f"Daily mean acoustic activity per recording for {site}_{plot} in {year}"
            )
            + xlab("Date")
            + ylab("Daily mean activity per recording (s)")
            + scale_x_datetime(labels=format_date_short)
            + theme_classic()
            + theme(axis_text_x=element_text(angle=45))
        )

        plts.append(plt)
        # plt_norm = (
        #     ggplot(
        #         data=df,
        #         mapping=aes("date", "trend_norm", color="type"),
        #     )
        #     + geom_line()
        # )
        # plots.append(plt_norm)

    else:
        print("No events detected for {}".format(plot["name"]))

save_as_pdf_pages(plts, res_dir / "all_plots.pdf")

agg_plts_dfs = {key: pd.concat(value) for key, value in agg_plts.items()}

for key, value in agg_plts_dfs.items():
    value.to_csv(res_dir / f"{key}_agg_trends.csv")
    # aplt = (
    #     ggplot(
    #         data=value,
    #         mapping=aes("date", "trend", color="type"),
    #     )
    #     + geom_line()
    #     + ggtitle(f"Daily mean acoustic activity per recording in {key}")
    #     + xlab("Date")
    #     + ylab("Daily mean activity per recording (s)")
    #     + scale_x_datetime(labels=format_date_short)
    #     + theme_classic()
    #     + theme(axis_text_x=element_text(angle=45))
    # ).save(res_dir / f"{key}_trends.png", width=10, height=8)


errors_df = pd.DataFrame(errors)
errors_df.to_csv(res_dir / "errors.csv")

#%%

for agg_file in res_dir.glob("*agg_trends.csv"):
    year = agg_file.stem.split("_")[0]
    df = pd.read_csv(agg_file)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    # aplt = (
    #     ggplot(
    #         data=df,
    #         mapping=aes("date", "trend", color="type"),
    #     )
    #     + geom_line()
    #     + geom_point(mapping=aes(y="total_duration"))
    #     + facet_wrap("site", scales="free")
    #     + ggtitle(f"Daily mean acoustic activity per recording in {year}")
    #     + xlab("Date")
    #     + ylab("Daily mean activity per recording (s)")
    #     + scale_x_datetime(labels=format_date_short)
    #     + theme_classic()
    #     + theme(axis_text_x=element_text(angle=45))
    # ).save(res_dir / f"trends_{year}_points_free.png", width=20, height=16)
    # aplt = (
    #     ggplot(
    #         data=df,
    #         mapping=aes("date", "trend_norm", color="type"),
    #     )
    #     + geom_line()
    #     + ggtitle(f"Daily mean acoustic activity per recording in {year}")
    #     + facet_wrap("site")
    #     + xlab("Date")
    #     + ylab("Daily mean activity per recording (s)")
    #     + scale_x_datetime(labels=format_date_short)
    #     + theme_classic()
    #     + theme(axis_text_x=element_text(angle=45))
    # ).save(res_dir / f"trends_norm_{year}.png", width=20, height=16)

    single_plots = []

    for site in df["site"].unique():
        tmp_df = df.loc[df.site == site]
        aplt = (
            ggplot(
                data=tmp_df,
                mapping=aes("date", "trend", color="type"),
            )
            + geom_line()
            # + geom_point(mapping=aes(y="total_duration"))
            + ggtitle(f"Daily mean acoustic activity per recording in {year}")
            + xlab("Date")
            + ylab("Daily mean activity per recording (s)")
            + scale_x_datetime(labels=format_date_short)
            + theme_classic()
            + theme(axis_text_x=element_text(angle=45))
        )
        single_plots.append(aplt)

    save_as_pdf_pages(single_plots, res_dir / f"single_plots_{year}.pdf")
