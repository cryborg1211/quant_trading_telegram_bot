import polars as pl
import numpy as np

df = pl.read_parquet("data/alpha360_features.parquet")
print("Total Rows:", len(df))

# Check for Nulls
null_count = df.null_count().sum().to_numpy().sum()
print("Total Nulls:", null_count)

# Check for Infs
numeric_df = df.select(pl.all().exclude(["date", "ticker"]))
numpy_data = numeric_df.to_numpy()
inf_count = np.isinf(numpy_data).sum()
print("Total Infs:", inf_count)

# Check Max/Min
print("Absolute Max:", np.nanmax(numpy_data))
print("Absolute Min:", np.nanmin(numpy_data))

# Sample targets
targets = df["target_return_1d"].to_numpy()
print("Target NaNs:", np.isnan(targets).sum())
print("Target Mean:", np.nanmean(targets))
print("Target Std:", np.nanstd(targets))
