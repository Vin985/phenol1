#%%

import pandas as pd
from pathlib import Path
from plotnine import *


src_path = Path("/home/vin/Doctorat/data/dl_training/raw/ArcticChecked/")

wav_files = list(src_path.rglob("*.WAV"))
tag_files = list(src_path.rglob("*.csv"))

df_presence = pd.DataFrame(
    {"file_name": [str(f) for f in wav_files], "has_bird": [0] * len(wav_files)}
)

splits = df_presence.file_name.str.split("/")

df_presence["year"] = splits.str[-4]
df_presence["site"] = splits.str[-3]
df_presence["plot"] = splits.str[-2]
df_presence["name"] = splits.str[-1].str.split(".").str[0]

print(df_presence)

tag_dfs = []

for row in df_presence.itertuples():
    tag_file = src_path / row.year / row.site / row.plot / (row.name + "-tags.csv")
    if tag_file.exists():
        tag_df = pd.read_csv(tag_file, skip_blank_lines=True)
        if not tag_df.empty:
            df_presence.loc[row.Index, "has_bird"] = 1
            tag_dfs.append(tag_df)

print(df_presence.has_bird)

#%%

stats = {}

stats["prop_has_bird"] = sum(df_presence.has_bird / df_presence.shape[0])

stats["prop_by_plot"] = df_presence.groupby("plot").agg(
    {"has_bird": lambda x: round(sum(x) / len(x) * 100, 2)}
)

stats["prop_by_year"] = df_presence.groupby("year").agg(
    {"has_bird": lambda x: round(sum(x) / len(x) * 100, 2)}
)

stats

#%%


ggplot(df_presence, aes("has_bird", fill="plot")) + geom_bar(position="dodge")
