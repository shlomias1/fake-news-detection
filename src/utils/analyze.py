import polars as pl

def analyze_columns(df: pl.DataFrame):
    for col in df.columns:
        null_count = df.select(pl.col(col).is_null().sum()).item()
        unique_values = df.select(pl.col(col).unique()).to_series().to_list()
        print(f"\nColumn: {col}")
        print(f"   - Missing values: {null_count}")
        print(f"   - Unique values: {unique_values[:20]}")
        if len(unique_values) > 20:
            print(f"   - ...and {len(unique_values) - 20} more")

def check_categories_columns(df: pl.DataFrame):
    print(f'type:\n{df["type"].unique().to_list()}')
    print(df.select(pl.col("type").value_counts()))
    print(f'category:\n{df["category"].unique().to_list()}')
    print(df.select(pl.col("category").value_counts()))
    
def check_missing_and_empty(df: pl.DataFrame, columns: list[str]):
    for col in columns:
        null_count = df.filter(pl.col(col).is_null()).height
        empty_count = df.filter((pl.col(col) == "") | (pl.col(col).str.strip_chars().is_in([""]))).height
        total = df.shape[0]
        print(f"\nColumn: {col}")
        print(f"Missing values (null): {null_count}")
        print(f"Empty values (''): {empty_count}")
        print(f"Valid values: {total - null_count - empty_count}")

def check_text_missing_but_url_exists(df: pl.DataFrame):
    df = df.with_columns([
        (
            (pl.col("text").is_null() | (pl.col("text") == "")) &            
            (~pl.col("url").is_null()) & (pl.col("url") != "")               
        ).alias("text_missing_but_url_exists")
    ])
    count = df.filter(pl.col("text_missing_but_url_exists") == True).height
    print(f"Number of rows where text is missing but url exists: {count}")