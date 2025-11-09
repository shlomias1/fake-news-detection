from newspaper import Article
import polars as pl

def scrape_article_text(url: str) -> str:
    try:
        if not url.startswith("http"):
            url = "https://" + url.lstrip(":/")

        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return None


def Running_Scraper(df: pl.DataFrame) -> pl.DataFrame:
    subset_df = df.filter(
        (pl.col("text").is_null() | (pl.col("text") == "")) &
        (~pl.col("url").is_null()) & (pl.col("url") != "")
    )
    urls = subset_df["url"].to_list()
    texts = [scrape_article_text(url) for url in urls]
    df_with_texts = subset_df.with_columns([
        pl.Series("scraped_text", texts)
    ])
    return df_with_texts

def Update_df_By_scrapper(df: pl.DataFrame, subset_df: pl.DataFrame) -> pl.DataFrame:
    df_updated = df.join(subset_df.select(["url", "scraped_text"]), on="url", how="left")
    df_updated = df_updated.with_columns([
        pl.when(pl.col("text").is_null() | (pl.col("text") == ""))
        .then(pl.col("scraped_text"))
        .otherwise(pl.col("text"))
        .alias("text")
    ])
    df_updated = df_updated.drop("scraped_text")
    return df_updated