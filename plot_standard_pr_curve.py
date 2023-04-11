#%%

import pandas as pd
from sklearn import metrics
from plotnine import *
import numpy as np

pr_df = pd.read_csv(
    "results/metrics/evaluation/20230330/154703_metrics_standard_pr_curve_stats.csv"
)
pr_df = pr_df.loc[pr_df.recall < 0.988]
pr_df = pr_df.append([{"precision": 1, "recall": 0}], ignore_index=True)
pr_df = pr_df.loc[:, ["precision", "recall"]].sort_values("precision")


pr_curve = ggplot(pr_df, aes(x="precision", y="recall")) + geom_point()

print(pr_curve)

print(metrics.auc(pr_df.recall, pr_df.precision))

print(pr_df.precision.mean())

ap = -np.sum(np.diff(pr_df.recall) * np.array(pr_df.precision)[:-1])
print(ap)

# %%
