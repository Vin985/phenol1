#%%
import pandas as pd
from pathlib import Path
import shutil
from mouffet import file_utils

file_list_path = Path(
    "/mnt/win/UMoncton/Doctorat/dev/phenol1/results/v2/events_to_explore/events_to_explore.csv"
)

tmp_dir = file_utils.ensure_path_exists("./events_files")

events_df = pd.read_csv(file_list_path)

events_df_files = events_df.recording_id.unique()

for tmp_file in events_df_files:
    shutil.copy(tmp_file, tmp_dir)
