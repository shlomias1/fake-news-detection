import preproccesing
import polars as pl
import data_io
import os
import utils.analyze as analyze
import utils.text_utils as text_utils

def pipeline():
    combined_df = preproccesing.merge_datasets()
    analyze.analyze_columns(combined_df)
    top_fake_words = text_utils.get_top_words(combined_df, 0)
    print(f"top fake words: {top_fake_words}")
    top_real_words = text_utils.get_top_words(combined_df, 1) 
    print(f"top real words: {top_real_words}")
    
if __name__ == "__main__":
    pipeline()
