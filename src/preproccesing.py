import pandas as pd
import polars as pl
import data_io
import utils.preprocessing_utils as preprocessing_utils
import utils.text_utils as text_utils
import os
import config
from tqdm import tqdm
tqdm.pandas()

def FakeNewsNet_processing():
    gossip_fake = data_io.load_csv(r'data/FakeNewsNet/gossipcop_fake.csv')
    gossip_real = data_io.load_csv(r'data/FakeNewsNet/gossipcop_real.csv')
    political_fake = data_io.load_csv(r'data/FakeNewsNet/politifact_fake.csv')
    political_real = data_io.load_csv(r'data/FakeNewsNet/politifact_real.csv')
    gossip_fake = preprocessing_utils.add_default_fields(gossip_fake, {'label': 0, 'category': 'gossip'})
    gossip_real = preprocessing_utils.add_default_fields(gossip_fake, {'label': 1, 'category': 'gossip'})
    political_fake = preprocessing_utils.add_default_fields(gossip_fake, {'label': 0, 'category': 'political'})
    political_real = preprocessing_utils.add_default_fields(gossip_fake, {'label': 1, 'category': 'political'})
    FakeNewsNet = pd.concat([gossip_fake, gossip_real, political_fake, political_real], ignore_index=True)
    FakeNewsNet = preprocessing_utils.rename_columns(FakeNewsNet, {'news_url': 'url'})
    FakeNewsNet.drop('tweet_ids', axis=1, inplace=True)
    FakeNewsNet['url'] = FakeNewsNet['url'].astype(str)
    FakeNewsNet['source'] = FakeNewsNet['url'].apply(preprocessing_utils.url_parser)
    FakeNewsNet['author'] = FakeNewsNet['title'].apply(preprocessing_utils.extract_author)
    FakeNewsNet['date_published'] = FakeNewsNet['url'].apply(preprocessing_utils.extract_date_from_url)
    default_fields = {'text': '', 'type': 'unknown','platform' : 'website','has_image' : 0, 'rating' : 0.5}
    FakeNewsNet = preprocessing_utils.add_default_fields(FakeNewsNet, default_fields)
    print("FakeNewsNet loaded")
    return pl.from_pandas(preprocessing_utils.reorder_df(FakeNewsNet))
    #return preprocessing_utils.reorder_df(FakeNewsNet)
    
def fake_true_dataset_processing():
    fake_news = data_io.load_csv(r'data/therealsampat/Fake.csv')
    true_news = data_io.load_csv(r'data/therealsampat/True.csv')
    fake_news["label"] = 0
    true_news["label"] = 1
    fake_true_dataset = pd.concat([fake_news, true_news], ignore_index=True)
    fake_true_dataset = preprocessing_utils.rename_columns(fake_true_dataset, {'subject': 'category','date': 'date_published'}) 
    fake_true_dataset['id'] = pd.Series(fake_true_dataset.index.astype(str)).apply(preprocessing_utils.generate_hash)
    fake_true_dataset['url'] = fake_true_dataset['text'].apply(preprocessing_utils.extract_url_from_text)
    default_fields = {'source' : '', 'author': 'unknown', 'type': 'unknown','platform' : 'website','has_image' : 0, 'rating' : 0.5}
    fake_true_dataset = preprocessing_utils.add_default_fields(fake_true_dataset, default_fields)
    print("fake_true_dataset loaded")
    return pl.from_pandas(preprocessing_utils.reorder_df(fake_true_dataset))
    #return preprocessing_utils.reorder_df(fake_true_dataset)

def fake_or_real_news_processing():
    fake_or_real_news = data_io.load_csv(r'data/fake_or_real_news.csv')
    if 'Unnamed: 0' in fake_or_real_news.columns:
        fake_or_real_news = fake_or_real_news.drop(columns=['Unnamed: 0'])
    fake_or_real_news['id'] = pd.Series(fake_or_real_news.index.astype(str)).apply(preprocessing_utils.generate_hash)
    fake_or_real_news['label'] = fake_or_real_news['label'].str.lower().map({'fake': 0, 'real': 1})
    fake_or_real_news['url'] = fake_or_real_news['text'].apply(preprocessing_utils.extract_url_from_text)
    default_fields = {'category': '','type': 'unknown','author': 'unknown','source': '','date_published': '','platform': 'website','has_image': 0,'rating': 0.5}
    fake_or_real_news = preprocessing_utils.add_default_fields(fake_or_real_news, default_fields)
    print("fake_or_real_news loaded")
    return pl.from_pandas(preprocessing_utils.reorder_df(fake_or_real_news))
    #return preprocessing_utils.reorder_df(fake_or_real_news)

def news_articales_processing():
    columns = ['author', 'published', 'title', 'text', 'site_url', 'type', 'label', 'hasImage']
    news_articles = data_io.load_csv(r'data/news_articles.csv')[columns]
    news_articles.rename(columns={'site_url': 'url'}, inplace=True)
    news_articles['id'] = pd.Series(news_articles.index.astype(str)).apply(preprocessing_utils.generate_hash)
    news_articles['label'] = news_articles['label'].str.lower().map({'fake': 0, 'real': 1})
    news_articles['source'] = news_articles['url'].apply(preprocessing_utils.url_parser)
    speaker_reliability = news_articles.groupby('author')['label'].mean().to_dict()
    news_articles['rating'] = news_articles['author'].map(speaker_reliability)
    news_articles['rating'] = news_articles['rating'].fillna(0.5)
    default_fields = {'category': 'articles','platform': 'website'}
    news_articles = preprocessing_utils.add_default_fields(news_articles, default_fields)
    print("news_articles loaded")
    return pl.from_pandas(preprocessing_utils.reorder_df(news_articles))
    #return preprocessing_utils.reorder_df(news_articles)

def liar_data_processing():
    test = data_io.load_liar_table(r'data/liar_data/test.tsv')
    train = data_io.load_liar_table(r'data/liar_data/train.tsv')
    valid = data_io.load_liar_table(r'data/liar_data/valid.tsv')
    liar_data = pd.concat([test, train, valid])
    liar_data['label'] = liar_data["label"].apply(lambda x: 0 if x in ["false", "pants-fire", "barely-true"] else 1)
    for col in ['barely_true_c', 'false_c', 'half_true_c', 'mostly_true_c', 'pants_on_fire_c']:
        liar_data[col] = liar_data[col].fillna(0)
    weights = {
        'pants_on_fire_c': -2,
        'false_c': -1.5,
        'barely_true_c': -1,
        'half_true_c': 0.5,
        'mostly_true_c': 1,
    }
    liar_data['rating'] = (
        liar_data['pants_on_fire_c'] * weights['pants_on_fire_c'] +
        liar_data['false_c'] * weights['false_c'] +
        liar_data['barely_true_c'] * weights['barely_true_c'] +
        liar_data['half_true_c'] * weights['half_true_c'] +
        liar_data['mostly_true_c'] * weights['mostly_true_c']
    )
    max_rating = liar_data['rating'].max()
    min_rating = liar_data['rating'].min()
    liar_data['rating'] = (liar_data['rating'] - min_rating) / (max_rating - min_rating)
    speaker_reliability = liar_data.groupby('speaker')['label'].mean().to_dict()
    liar_data['speaker_reliability'] = liar_data['speaker'].map(speaker_reliability)
    liar_data['speaker_reliability'] = liar_data['speaker_reliability'].fillna(0.5)
    liar_data['rating'] = liar_data['rating'] * 0.7 + liar_data['speaker_reliability'] * 0.3
    liar_data['subject'] = liar_data['subject'].fillna('unknown')
    liar_data = preprocessing_utils.rename_columns(liar_data, {'subject': 'title','statement': 'text'})
    liar_data['url'] = liar_data['text'].apply(preprocessing_utils.extract_url_from_text)
    liar_data['author'] = liar_data['title'].apply(preprocessing_utils.extract_author)
    default_fields = {'source': '','date_published': '', 'platform' : 'website', 'has_image' : 0, 'category' : '', 'type' : 'unknown'}
    liar_data = preprocessing_utils.add_default_fields(liar_data, default_fields)
    print("liar_data loaded")
    return pl.from_pandas(preprocessing_utils.reorder_df(liar_data))
    #return preprocessing_utils.reorder_df(liar_data)

def Fake_Stories_processing():
    Fake_Stories = data_io.load_xlsx('data/Fake News Stories.xlsx')
    Fake_Stories["category"] = "stories"
    Fake_Stories["label"] = 0
    Fake_Stories = preprocessing_utils.rename_columns(Fake_Stories, {'Fake or Satire?': 'title','URL of article': 'url'})
    Fake_Stories['title'] = Fake_Stories['url'].apply(preprocessing_utils.extract_title_from_url)
    Fake_Stories['source'] = Fake_Stories['url'].apply(preprocessing_utils.url_parser)
    default_fields = {'author': 'unknown', 'date_published': '', 'platform': 'website', 'has_image': 0, 'rating': 0.5}
    Fake_Stories = preprocessing_utils.add_default_fields(Fake_Stories, default_fields)    
    print("Fake_Stories loaded")
    return pl.from_pandas(preprocessing_utils.reorder_df(Fake_Stories))

def YoutubeVideo_processing():
    columns = ['vid_url', 'vid_title', 'aria-label', 'normalized_annotation', 'topic']
    Youtube_fake_news = data_io.load_csv(r'data/all_results.csv')[columns]
    Youtube_fake_news = Youtube_fake_news[Youtube_fake_news['normalized_annotation'] != 0] 
    Youtube_fake_news['label'] = Youtube_fake_news['normalized_annotation'].map({-1: 0, 1: 1})
    Youtube_fake_news = preprocessing_utils.rename_columns(Youtube_fake_news, {'vid_title': 'title','vid_url': 'url', 'topic': 'type'})
    default_fields = {'platform': 'youtube', 'text': '', 'date_published': '', 'author': 'unknown', 'has_image': 0, 'category' : 'Videos'}
    Youtube_fake_news = preprocessing_utils.add_default_fields(Youtube_fake_news, default_fields)    
    Youtube_fake_news['id'] = pd.Series(Youtube_fake_news.index.astype(str)).apply(preprocessing_utils.generate_hash)
    Youtube_fake_news['source'] = Youtube_fake_news['aria-label'].apply(preprocessing_utils.extract_channel_name)
    Youtube_fake_news['rating'] = Youtube_fake_news['source'].apply(preprocessing_utils.classify_channel_risk)
    Youtube_fake_news = Youtube_fake_news.drop_duplicates()
    print("Youtube_fake_news loaded")
    return pl.from_pandas(preprocessing_utils.reorder_df(Youtube_fake_news))

def BuzzFead_Facebook_processing():
    BuzzFead_Facebook = data_io.load_csv(r'data/facebook-fact-check.csv')
    default_fields = {'platform': 'facebook', 'has_image' : 0, 'text' : '', 'author' : "unknown", 'type' : 'unknown', 'title' : ''}
    BuzzFead_Facebook = preprocessing_utils.add_default_fields(BuzzFead_Facebook, default_fields)  
    BuzzFead_Facebook = preprocessing_utils.rename_columns(BuzzFead_Facebook, {'Category': 'category','Post URL': 'url', 'Page': 'source', 'Date Published': 'date_published'})
    BuzzFead_Facebook['id'] = pd.Series(BuzzFead_Facebook.index.astype(str)).apply(preprocessing_utils.generate_hash)
    map_rating = {"mostly true": 1.0,
                "mixture of true and false": 0.5,
                "no factual content": 0.25,
                "mostly false": 0.0}
    BuzzFead_Facebook['rating'] = BuzzFead_Facebook['Rating'].str.lower().map(map_rating)
    BuzzFead_Facebook['label'] = (BuzzFead_Facebook['Rating'].str.lower() == 'mostly true').astype(int)
    print("BuzzFead_Facebook loaded")
    return pl.from_pandas(preprocessing_utils.reorder_df(BuzzFead_Facebook))

def _BanFakeNews_processing():
    merged_file_path = 'data/BanFakeNews/banfakenews_merged_full.csv'
    if not os.path.exists(merged_file_path):
        BanFakeNews_Authentic = data_io.load_csv(r'data/BanFakeNews/Authentic.csv')
        BanFakeNews_FakeNews = data_io.load_csv(r'data/BanFakeNews/FakeNews.csv')
        BanFakeNews_labeledAuthentic = data_io.load_csv(r'data/BanFakeNews/LabeledAuthentic.csv')
        BanFakeNews_labeledFakeNews = data_io.load_csv(r'data/BanFakeNews/LabeledFake.csv')
        combined_df = pd.concat([
            BanFakeNews_Authentic,
            BanFakeNews_FakeNews,
            BanFakeNews_labeledAuthentic,
            BanFakeNews_labeledFakeNews
        ], sort=False, ignore_index=True)
        combined_df.to_csv(merged_file_path, index=False)
    columns = ['domain', 'date', 'category', 'headline', 'content', 'label','F-type']
    BanFakeNews = data_io.load_csv(merged_file_path)[columns]
    BanFakeNews['id'] = pd.Series(BanFakeNews.index.astype(str)).apply(preprocessing_utils.generate_hash)
    BanFakeNews = preprocessing_utils.rename_columns(BanFakeNews, {'domain': 'source','date': 'date_published', 'headline': 'title', 'content': 'text'})
    BanFakeNews.loc[BanFakeNews['F-type'].isna() & (BanFakeNews['label'] == 0), 'F-type'] = 'Fake'
    BanFakeNews.loc[BanFakeNews['F-type'].isna() & (BanFakeNews['label'] == 1), 'F-type'] = 'unknown'
    BanFakeNews.rename(columns={'F-type': 'type'}, inplace=True)    
    BanFakeNews['title_lang'] = BanFakeNews['title'].progress_apply(
            lambda x: text_utils.detect_lang(x) if pd.notnull(x) and str(x).strip() else None
        )    
    BanFakeNews['text_lang'] = BanFakeNews['text'].progress_apply(
            lambda x: text_utils.detect_lang(x) if pd.notnull(x) and str(x).strip() else None
        )
    BanFakeNews = text_utils.translate(BanFakeNews, "title", "title_lang", "title_translated", target_lang='en')
    BanFakeNews = text_utils.translate(BanFakeNews, "text", "text_lang", "text_translated", target_lang='en')
    BanFakeNews.drop(['text_lang','title_lang','title','text'], axis=1, inplace=True)
    BanFakeNews = preprocessing_utils.rename_columns(BanFakeNews, {'title_translated': 'title','text_translated': 'text'})
    default_fields = {'platform': 'website', 'has_image': 0, 'author': 'unknown', 'url': '', 'rating': 0.5}
    BanFakeNews = preprocessing_utils.add_default_fields(BanFakeNews, default_fields)  
    BanFakeNews.to_csv('data/BanFakeNews/All_BanFakeNews.csv')
    print("BanFakeNews saved")
    return pl.from_pandas(preprocessing_utils.reorder_df(BanFakeNews))
    #return preprocessing_utils.reorder_df(BanFakeNews)

def BanFakeNews_processing():
    if not os.path.exists('data/BanFakeNews/All_BanFakeNews.csv'):
        df = _BanFakeNews_processing()
        return df
    df = data_io.load_csv(r'data/BanFakeNews/All_BanFakeNews.csv', "polars")
    return df

def FakeNewsBot_processing():
    FakeNewsBot = data_io.load_csv(r'data/FakeNewsBot.csv')
    FakeNewsBot['id'] = pd.Series(FakeNewsBot.index.astype(str)).apply(preprocessing_utils.generate_hash)
    author_reliability = FakeNewsBot.groupby('author')['label'].mean().to_dict()
    FakeNewsBot['rating'] = FakeNewsBot['author'].map(author_reliability)
    FakeNewsBot['rating'] = FakeNewsBot['rating'].fillna(0.5)
    FakeNewsBot['author'] = FakeNewsBot['author'].fillna('unknown')
    FakeNewsBot['title'] = FakeNewsBot['title'].fillna('')
    FakeNewsBot['text'] = FakeNewsBot['text'].fillna('')
    FakeNewsBot['url'] = FakeNewsBot['text'].apply(preprocessing_utils.extract_url_from_text)
    default_fields = {'category': 'unknown', 'type': 'unknown', 'platform': 'website', 'has_image': 0, 'date_published': '', 'source': ''}
    FakeNewsBot = preprocessing_utils.add_default_fields(FakeNewsBot, default_fields) 
    print("FakeNewsBot loaded")
    return pl.from_pandas(preprocessing_utils.reorder_df(FakeNewsBot))

def opensources_fake_news_processing():
    opensources_fake_news = data_io.load_csv("data/FakeNewsDataset/opensources_fake_news_cleaned.csv","polars")
    opensources_fake_news = opensources_fake_news.with_columns(
        pl.Series("id", [preprocessing_utils.generate_hash(str(i)) for i in range(opensources_fake_news.height)])
    )
    opensources_fake_news = opensources_fake_news.rename({
            'domain': 'source',
            'scraped_at': 'date_published',
            'authors': 'author',
            'content': 'text'
        })
    opensources_fake_news = opensources_fake_news.with_columns([
        opensources_fake_news["type"].map_elements(preprocessing_utils._label_verdict, return_dtype=pl.Int64).alias("label")
    ])
    opensources_fake_news = opensources_fake_news.filter(pl.col("label") != -1)
    author_reliability = (
        opensources_fake_news
        .group_by("author")
        .agg(pl.col("label").mean().alias("author_avg"))
    )
    opensources_fake_news = (
        opensources_fake_news
        .join(author_reliability, on="author", how="left")
        .with_columns(
            pl.col("author_avg").fill_null(0.5).alias("rating")
        )
        .drop("author_avg") 
    )
    opensources_fake_news = opensources_fake_news.with_columns([
        pl.col("author").fill_null("unknown"),
        pl.col("title").fill_null(0.5)

    ])
    opensources_fake_news = opensources_fake_news.with_columns([
        pl.col("text").map_elements(preprocessing_utils.extract_url_from_text, return_dtype=pl.Utf8).alias("url")
    ])
    opensources_fake_news = opensources_fake_news.with_columns([
        pl.lit("unknow").alias("category"),
        pl.lit("website").alias("platform"),
        pl.lit(0).alias("has_image")
    ])
    print("opensources_fake_news loaded")
    return opensources_fake_news.select(config.column_order)

def normalize_column(df: pl.DataFrame, col: str, mapping: dict, default="unknown") -> pl.DataFrame:
    return df.with_columns([
        pl.col(col)
        .cast(pl.Utf8)
        .str.strip_chars()
        .str.to_lowercase()
        .map_elements(lambda val: mapping.get(val, default))
        .alias(col)
    ])

def merge_datasets():
    merged_file_path = "/home/shlomias/fake_news_detection/data/fake_news_combined_dataset/fake_news_combined_dataset.csv"
    zip_path = "/home/shlomias/fake_news_detection/data/fake_news_combined_dataset.zip"
    #data_io.extract_zip(zip_path, merged_file_path)
    # if not os.path.exists(merged_file_path):
    #     datasets = [
    #         FakeNewsNet_processing(),
    #         fake_true_dataset_processing(),
    #         fake_or_real_news_processing(),
    #         news_articales_processing(),
    #         liar_data_processing(),
    #         Fake_Stories_processing(),
    #         YoutubeVideo_processing(),
    #         BuzzFead_Facebook_processing(),
    #         BanFakeNews_processing(),
    #         FakeNewsBot_processing(),
    #         opensources_fake_news_processing()
    #     ]
    #     combined_df = pl.concat(datasets, how="diagonal_relaxed")
    #     combined_df.write_csv(merged_file_path)
    df = pl.read_csv(merged_file_path, columns=config.column_order)
    df = df.filter(pl.col("label").is_not_null())
    df = normalize_column(df, "type", config.TYPE_MAPPING)
    df = normalize_column(df, "category", config.CATEGORY_MAPPING)
    df = df.filter(
        (~pl.col("text").is_null()) & (pl.col("text") != "")
    )
    return df