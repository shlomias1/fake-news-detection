import preproccesing
import polars as pl
import data_io
import os

def analyze_columns(df: pl.DataFrame):
    for col in df.columns:
        null_count = df.select(pl.col(col).is_null().sum()).item()
        unique_values = df.select(pl.col(col).unique()).to_series().to_list()
        print(f"\n🔹 Column: {col}")
        print(f"   - Missing values: {null_count}")
        print(f"   - Unique values: {unique_values[:20]}")
        if len(unique_values) > 20:
            print(f"   - ...and {len(unique_values) - 20} more")

def pipeline():
    combined_df = preproccesing.merge_datasets()
    analyze_columns(combined_df)
    
if __name__ == "__main__":
    pipeline()
