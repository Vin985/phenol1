#%%

import pandas as pd

from pathlib import Path

sum_path = Path("/mnt/win/UMoncton/Doctorat/results/summaries")

all_summaries = []
for summary in sum_path.glob("classes_*.csv"):
    all_summaries.append(pd.read_csv(summary))

all_df = pd.concat(all_summaries)

not_bird = [
    "Insect",
    "Animal",
    "amphibian",
    "animal",
    "invertebrate",
    "russling leaves (animal)",
    "wing beats",
    "wings beating",
    "bat",
    "Wing Beats",
]

no_bird = all_df.loc[all_df.tag.isin(not_bird)]

total_size = all_df.tag_size.sum()
no_bird_size = no_bird.tag_size.sum()

prop_no_bird = no_bird_size / total_size * 100

print(
    f"Annotations not birds: {no_bird_size}. Total: {total_size}. Prop: {prop_no_bird}"
)

# %%
