#%%
import pandas as pd
from pathlib import Path


old_root = Path("/home/sylvain/data")
dataset_path = Path("/mnt/win/UMoncton/Doctorat/data/dl_training/datasets")
new_root = Path("/mnt/win/UMoncton/Doctorat/data/dl_training")


for path in dataset_path.iterdir():
    if path.is_dir() and (str(path).endswith("_final") or str(path).endswith("_all")):
        for csv_file in path.glob("*.csv"):
            df = pd.read_csv(csv_file, header=None)
            df.iloc[:, 0] = df.iloc[:, 0].str.replace(str(old_root), str(new_root))
            print(df)
            df.to_csv(csv_file, index=False, header=None)

# %%
