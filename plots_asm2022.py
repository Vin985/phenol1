#%%


from pathlib import Path
import pandas as pd
from plotnine import *

# src_dir = Path("/mnt/win/UMoncton/Doctorat/dev/phenol1/results/v2")
src_dir = Path("/mnt/win/UMoncton/Doctorat/results/predictions_v2")
# dest_dir = Path("/mnt/win/UMoncton/Doctorat/presentations/ArcticNet 2022/figs")
dest_dir = src_dir / "plots"

text_size = 16
title_size = 18
legend_title_margin = {"b": 15}
activity_threshold = 0.75
method = "direct"
trend = "trend"


def format_date_short(dates):
    if (dates[len(dates) - 1] - dates[0]).days > 1:
        return [x.strftime("%d") for x in dates]
    else:
        return [x.strftime("%H:%M") for x in dates]


def read_trends(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    return df


###* IGLOOLIK 2019 INTRA SITE PLOTS
df = read_trends(src_dir / f"2018_agg_trends_{method}_{activity_threshold}.csv")
tmp_df = df.loc[df.site == "IGLO"].reset_index()
tmp_df = tmp_df.loc[~tmp_df.type.isin(["IGLO_H", "IGLO_10"])].reset_index()

cbbPalette = [
    "#56B4E9",
    "#E69F00",
    # "#000000",
    "#4D4D4D",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#F0E442",
]

# "#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"

aplt = (
    ggplot(
        data=tmp_df,
        mapping=aes("date", trend, color="type"),
    )
    + geom_line(size=2)
    # + scale_color_brewer(type="qual", palette="Accent")
    + scale_colour_manual(
        values=cbbPalette,
        name="Plot name",
        # labels=[
        #     "Plot A",
        #     "Plot B",
        #     "Plot D",
        #     "Plot E",
        #     "Plot F",
        # ],
        # guide=False,
    )
    # + geom_point(mapping=aes(y="total_duration"))
    # + ggtitle(
    #     f"Daily mean acoustic activity per recording between plots in Igloolik in 2018"
    # )
    + xlab("Date")
    + ylab("Daily mean activity per recording (s)")
    + scale_x_datetime(labels=format_date_short)
    + theme_classic()
    + theme(
        # axis_text_x=element_text(angle=45),
        text=element_text(weight="bold"),
        axis_text=element_text(size=text_size),
        axis_title=element_text(size=title_size),
        title=element_text(size=title_size),
        axis_line=element_line(size=2),
        legend_text=element_text(size=text_size),
        legend_title=element_text(margin=legend_title_margin),
        # panel_background=element_rect(fill="#FFFFFF00"),  # transparent panel bg
        # plot_background=element_rect(fill="#FFFFFF00", color=None),
    )
).save(
    dest_dir / f"iglo_intra_noleg_{method}_{activity_threshold}_{trend}.png",
    width=12,
    height=7,
)

print(aplt)
#%%


##* All sites 2019
df = read_trends(src_dir / f"2018_agg_trends_{method}_{activity_threshold}.csv")
tmp_df = df.groupby(["date", "site"]).mean().reset_index()
tmp_df = tmp_df.loc[
    tmp_df.site.isin(["CORI", "CHUR", "ALRT", "EABA", "SVAL", "ZACK", "PBPS"])
].reset_index()

tmp_df = tmp_df.loc[
    (tmp_df.date > "2018-06-03") & (tmp_df.date < "2018-08-01")
].reset_index()

aplt = (
    ggplot(
        data=tmp_df,
        mapping=aes("date", trend, color="site"),
    )
    + geom_line(size=2)
    # + scale_color_brewer(type="qual", palette="Accent")
    + scale_colour_manual(
        values=cbbPalette,
        name="Site",
        labels=[
            "Alert",
            "Churchill",
            "Colville River",
            "East Bay",
            "Nanuit Itillinga",
            "Longyearbyen",
            "Zackenberg",
        ],
        guide=False,
    )
    # + geom_point(mapping=aes(y="total_duration"))
    # + ggtitle(f"Daily mean acoustic activity per recording between sites in 2018")
    + xlab("Date")
    + ylab("Daily mean activity per recording (s)")
    + scale_x_datetime(labels=format_date_short)
    # + facet_wrap("site", nrow=2)
    + theme_classic()
    + theme(
        # axis_text_x=element_text(angle=45),
        text=element_text(weight="bold"),
        axis_text=element_text(size=text_size),
        axis_title=element_text(size=title_size),
        title=element_text(size=title_size),
        axis_line=element_line(size=2),
        legend_text=element_text(size=text_size),
        legend_title=element_text(margin=legend_title_margin),
        # panel_background=element_rect(fill="#FFFFFF00"),  # transparent panel bg
        # plot_background=element_rect(fill="#FFFFFF00", color=None),
    )
).save(
    dest_dir / f"all_subset_2018_noleg_{method}_{activity_threshold}_{trend}.png",
    width=12,
    height=7,
)

#%%

##** Barrow Inter years


##* All sites 2019
tmp = []
for agg_file in src_dir.glob(f"*agg_trends_{method}_{activity_threshold}.csv"):
    year = agg_file.stem.split("_")[0]
    df = pd.read_csv(agg_file)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df["julian"] = df.date.dt.dayofyear
    df["year"] = int(year)
    if year == "2020":
        df = df.loc[df.julian > 171]

    # tmp.append(df.loc[df.type == "BARW_0"])
    tmp.append(df.loc[df.site == "BARW"])

tmp_df = pd.concat(tmp)
tmp_df2 = tmp_df.copy()
tmp_df = tmp_df.groupby(["year", "julian"]).mean().reset_index()

tmp_df = tmp_df.astype({"year": "category"})


# tmp_df = tmp_df.loc[
#     (tmp_df.date > "2018-05-31") & (tmp_df.date < "2018-08-01")
# ].reset_index()

aplt = (
    ggplot(
        data=tmp_df,
        mapping=aes("julian", trend, color="year"),
    )
    + geom_line(size=2)
    # + scale_color_brewer(type="qual", palette="Accent")
    + scale_colour_manual(
        values=cbbPalette[0:3],
        name="Year",
        guide=False,
        # labels=[
        # ],
    )
    # + geom_point(mapping=aes(y="total_duration"))
    # + ggtitle(f"Daily mean acoustic activity among years at Utqiaġvik")
    + xlab("Julian day")
    + ylab("Daily mean activity per recording (s)")
    # + scale_x_datetime(labels=format_date_short)
    # + facet_wrap("site", nrow=2)
    + theme_classic()
    + theme(
        # axis_text_x=element_text(angle=45),
        text=element_text(weight="bold"),
        axis_text=element_text(size=text_size),
        axis_title=element_text(size=title_size),
        title=element_text(size=title_size),
        axis_line=element_line(size=2),
        legend_text=element_text(size=text_size),
        legend_title=element_text(margin=legend_title_margin),
        # panel_background=element_rect(fill="#FFFFFF00"),  # transparent panel bg
        # plot_background=element_rect(fill="#FFFFFF00", color=None),
    )
).save(
    dest_dir / f"BARW_all_years_noleg_{method}_{activity_threshold}_{trend}.png",
    width=12,
    height=7,
)

# %%
