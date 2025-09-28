from config import STOP_ALL, HEBREW_NIKKUD, NON_LETTERS_KEEP_HE, URL_RE, EMAIL_RE, CLICKBAIT_RE
import pandas as pd
from urllib.parse import urlparse
import polars as pl

def _count_runs_3plus(s: str) -> int:
    if not s:
        return 0
    total = 0
    run = 1
    prev = s[0]
    for ch in s[1:]:
        if ch == prev:
            run += 1
            if run == 3:
                total += 1
        else:
            run = 1
            prev = ch
    return total

def build_text_features(
    df: pl.DataFrame,
    title_col: str = "title",
    text_col: str = "text",
    drop_intermediate: bool = True
) -> pl.DataFrame:
    # 1) נורמליזציה קלה
    df = df.with_columns([
        pl.col(title_col).cast(pl.Utf8).fill_null("").alias("title_raw"),
        pl.col(text_col).cast(pl.Utf8).fill_null("").alias("text_raw"),
    ]).with_columns([
        pl.col("title_raw")
          .str.replace_all(HEBREW_NIKKUD, "")
          .str.replace_all(r"\s+", " ")
          .str.strip_chars()
          .alias("title_norm"),
        pl.col("text_raw")
          .str.replace_all(HEBREW_NIKKUD, "")
          .str.replace_all(r"\s+", " ")
          .str.strip_chars()
          .alias("text_norm"),
    ])

    # 2) טוקניזציה
    df = df.with_columns([
        pl.col("title_norm")
          .str.replace_all(NON_LETTERS_KEEP_HE, " ")
          .str.to_lowercase()
          .str.replace_all(r"\s+", " ")
          .str.strip_chars()
          .str.split(by=" ")
          .alias("title_tokens"),
        pl.col("text_norm")
          .str.replace_all(NON_LETTERS_KEEP_HE, " ")
          .str.to_lowercase()
          .str.replace_all(r"\s+", " ")
          .str.strip_chars()
          .str.split(by=" ")
          .alias("text_tokens"),
    ])

    # 3) פיצ'רים בסיסיים
    df = df.with_columns([
        pl.col("title_norm").str.len_chars().alias("title_n_chars"),
        pl.col("text_norm").str.len_chars().alias("text_n_chars"),

        pl.col("title_tokens").list.len().alias("title_n_tokens"),
        pl.col("text_tokens").list.len().alias("text_n_tokens"),

        pl.col("title_tokens").list.unique().list.len().alias("title_n_unique"),
        pl.col("text_tokens").list.unique().list.len().alias("text_n_unique"),

        # פיסוק/תבניות
        pl.col("title_norm").str.count_matches(r"[!?]").alias("title_exc_qm"),
        pl.col("text_norm").str.count_matches(r"[!?]").alias("text_exc_qm"),
        pl.col("text_norm").str.count_matches(r"\.\.\.+").alias("text_ellipsis"),
        pl.col("text_norm").str.count_matches(r"!!+").alias("text_multi_exclaim"),
        pl.col("text_norm").str.count_matches(r"\?\?+").alias("text_multi_qmark"),
        pl.col("text_norm").map_elements(_count_runs_3plus).alias("text_char_repeat_3plus"),

        # ספרות/מטבע
        pl.col("text_norm").str.count_matches(r"\d").alias("text_n_digits"),
        pl.col("text_norm").str.count_matches(r"\b\d+(\.\d+)?\b").alias("text_n_numbers"),
        pl.col("text_norm").str.count_matches(r"[$₪€£]").alias("text_currency"),

        # URLs / אימיילים
        pl.col("text_norm").str.count_matches(URL_RE).alias("text_url_count"),
        pl.col("text_norm").str.count_matches(EMAIL_RE).alias("text_email_count"),

        # “קליקבייט”
        pl.col("title_norm").str.count_matches(CLICKBAIT_RE).alias("title_clickbait_hits"),
        pl.col("text_norm").str.count_matches(CLICKBAIT_RE).alias("text_clickbait_hits"),

        # לטינית/אותיות גדולות
        pl.col("text_norm").str.count_matches(r"[A-Za-z]").alias("text_latin_letters"),
        pl.col("text_norm").str.count_matches(r"[A-Z]").alias("text_upper_letters"),

        # יחס עברית/לטינית
        pl.col("text_norm").str.count_matches(r"[\u0590-\u05FF]").alias("text_hebrew_chars"),
        pl.col("text_norm").str.count_matches(r"[A-Za-z]").alias("text_latin_chars"),

        # אומדן משפטים
        pl.max_horizontal([
            pl.lit(1),
            pl.col("text_norm").str.count_matches(r"[.!?]+")
        ]).alias("text_n_sentences_est"),
    ])

    # 4) יחסים
    df = df.with_columns([
        (pl.col("title_n_unique") / pl.col("title_n_tokens").clip(lower_bound=1)).alias("title_ttr"),
        (pl.col("text_n_unique")  / pl.col("text_n_tokens").clip(lower_bound=1)).alias("text_ttr"),

        (pl.col("text_upper_letters") / pl.col("text_latin_letters").clip(lower_bound=1)).alias("text_upper_ratio"),
        (pl.col("text_hebrew_chars") / pl.col("text_n_chars").clip(lower_bound=1)).alias("text_hebrew_char_ratio"),
        (pl.col("text_latin_chars")  / pl.col("text_n_chars").clip(lower_bound=1)).alias("text_latin_char_ratio"),

        (pl.col("text_n_tokens") / pl.col("text_n_sentences_est").clip(lower_bound=1)).alias("text_avg_tokens_per_sentence"),

        (pl.col("title_n_chars") / pl.col("text_n_chars").clip(lower_bound=1)).alias("title_text_char_ratio"),
        (pl.col("title_n_tokens") / pl.col("text_n_tokens").clip(lower_bound=1)).alias("title_text_token_ratio"),

        (pl.col("text_n_digits") / pl.col("text_n_chars").clip(lower_bound=1)).alias("text_digit_char_ratio"),
        (pl.col("text_n_numbers") / pl.col("text_n_tokens").clip(lower_bound=1)).alias("text_number_token_ratio"),
        (pl.col("text_exc_qm") / pl.col("text_n_tokens").clip(lower_bound=1)).alias("text_exclaim_qm_per_token"),
    ])

    # 5) Jaccard Title↔Text
    df = df.with_columns([
        pl.struct(["title_tokens", "text_tokens"]).map_elements(
            lambda s: (
                lambda a, b: (len(a & b) / max(1, len(a | b)))
            )(set(s["title_tokens"]), set(s["text_tokens"]))
        ).alias("title_text_jaccard")
    ])

    # 6) אורך מילה ממוצע
    df = df.with_columns([
        pl.when(pl.col("text_n_tokens") > 0)
          .then(
              pl.col("text_tokens").list.eval(pl.element().str.len_chars()).list.sum()
              / pl.col("text_n_tokens")
          ).otherwise(0.0).alias("text_avg_word_len"),

        pl.when(pl.col("title_n_tokens") > 0)
          .then(
              pl.col("title_tokens").list.eval(pl.element().str.len_chars()).list.sum()
              / pl.col("title_n_tokens")
          ).otherwise(0.0).alias("title_avg_word_len"),
    ])

    if drop_intermediate:
        df = df.drop([
            "title_raw","text_raw",
            "title_norm","text_norm",
            "title_tokens","text_tokens",
            "text_latin_letters","text_upper_letters",
            "text_hebrew_chars","text_latin_chars",
        ])
    return df

def add_without_stopwords(df: pl.DataFrame) -> pl.DataFrame:
    # (א) יצירת טוקנים ללא מילות עצירה
    df = df.with_columns([
        pl.col("title_tokens")
          .list.eval(pl.when(~pl.element().is_in(STOP_ALL)).then(pl.element()).otherwise(None))
          .list.drop_nulls()
          .alias("title_tokens_ns"),
        pl.col("text_tokens")
          .list.eval(pl.when(~pl.element().is_in(STOP_ALL)).then(pl.element()).otherwise(None))
          .list.drop_nulls()
          .alias("text_tokens_ns"),
    ])

    # (ב) ספירות וייחודיות (יוצר עמודות חדשות)
    df = df.with_columns([
        pl.col("title_tokens_ns").list.len().alias("title_n_tokens_ns"),
        pl.col("text_tokens_ns").list.len().alias("text_n_tokens_ns"),
        pl.col("title_tokens_ns").list.unique().list.len().alias("title_n_unique_ns"),
        pl.col("text_tokens_ns").list.unique().list.len().alias("text_n_unique_ns"),
    ])

    df = df.with_columns([
        (pl.col("title_n_unique_ns") / pl.col("title_n_tokens_ns").clip(lower_bound=1)).alias("title_ttr_ns"),
        (pl.col("text_n_unique_ns")  / pl.col("text_n_tokens_ns").clip(lower_bound=1)).alias("text_ttr_ns"),
        (1 - (pl.col("text_n_tokens_ns") / pl.col("text_n_tokens").clip(lower_bound=1))).alias("text_stopword_ratio"),
        (1 - (pl.col("title_n_tokens_ns") / pl.col("title_n_tokens").clip(lower_bound=1))).alias("title_stopword_ratio"),
    ])

    df = df.with_columns([
        pl.struct(["title_tokens_ns","text_tokens_ns"]).map_elements(
            lambda s: (
                lambda a, b: (len(a & b) / max(1, len(a | b)))
            )(set(s["title_tokens_ns"]), set(s["text_tokens_ns"]))
        , return_dtype=pl.Float64).alias("title_text_jaccard_ns")
    ])

    df = df.with_columns([
        pl.col("title_tokens_ns").list.join(" ").alias("title_ns_text"),
        pl.col("text_tokens_ns").list.join(" ").alias("text_ns_text"),
        (pl.col("title_tokens_ns").list.join(" ") + pl.lit(" ") + pl.col("text_tokens_ns").list.join(" ")).alias("titleplus_text_ns")
    ])
    return df

def feat_pipeline(df):
    df_feat = build_text_features(df, drop_intermediate=False)  
    df_feat = add_without_stopwords(df_feat)
    drop_text_cols = [
        "title","text",
        "title_raw","text_raw",
        "title_norm","text_norm",
        "title_tokens","text_tokens",
        "title_tokens_ns","text_tokens_ns",
        "text_latin_letters","text_upper_letters",
        "text_hebrew_chars","text_latin_chars",
    ]
    keep_string_cols = {"title_ns_text","text_ns_text","titleplus_text_ns"}
    drop_final = [c for c in drop_text_cols if (c in df_feat.columns and c not in keep_string_cols)]
    df_feat = (
        df_feat
        .select(pl.all().exclude(pl.List))  
        .drop(drop_final)                  
        .rechunk()
    )
    df_feat.write_csv("/home/shlomias/fake_news_detection/data/df_feat.csv")
    return df_feat