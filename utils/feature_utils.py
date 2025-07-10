import config
import pandas as pd
from urllib.parse import urlparse

def detect_platform_risk(platform):
    if pd.isna(platform):
        return 1  
    platform = platform.lower()
    if any(p in platform for p in config.high_risk):
        return 2
    elif any(p in platform for p in config.low_risk):
        return 0
    else:
        return 1

def keyword_flags(text):
    if pd.isnull(text):
        return 0
    text = text.lower()
    return int(any(kw in text for kw in config.KEYWORDS))

def add_keyword_features(df):
    df['title_has_keyword'] = df['title'].apply(keyword_flags)
    df['text_has_keyword'] = df['text'].apply(keyword_flags)
    return df

def extract_domain_extension(url):
    if pd.isnull(url):
        return ''
    try:
        netloc = urlparse(url).netloc
        return netloc.split('.')[-1]
    except:
        return ''

def domain_risk(ext):
    risky = ['co', 'xyz', 'info', 'top']
    trusted = ['com', 'org', 'edu', 'gov', 'net']
    if ext in risky:
        return 2
    elif ext in trusted:
        return 0
    else:
        return 1

def add_domain_features(df):
    df['domain_ext'] = df['url'].apply(extract_domain_extension)
    df['domain_risk'] = df['domain_ext'].apply(domain_risk)
    return df

def one_hot_encode_columns(df, columns):
    return pd.get_dummies(df, columns=columns, prefix=columns)

def map_super_category(cat):
    if pd.isnull(cat):
        return 'other'
    cat = cat.lower()
    return config.CATEGORY_MAPPING.get(cat, 'other')

def enrich_features(df):
    df = add_keyword_features(df)
    df = add_domain_features(df)
    df = one_hot_encode_columns(df, ['category', 'type'])
    df['super_category'] = df['category'].apply(map_super_category)
    return df