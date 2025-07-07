import config
import pandas as pd
from urllib.parse import urlparse
import nltk
from nltk.corpus import stopwords

# להוסיף עמודת שפה
# להוסיף עמודת תרגו לעברית של כותרת ושל תוכן
# key words - בדיקה איזה מילים חוזרות על עצמם בפייק ניוז

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def remove_stop_words(text):
    if pd.isnull(text):
        return ''
    words = text.split()
    filtered = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered)

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

def remove_hebrew_stop_words(text):
    if pd.isnull(text):
        return ''
    words = text.split()
    filtered = [w for w in words if w not in config.hebrew_stop_words]
    return ' '.join(filtered)

df['text_without_stopwords'] = df['text'].apply(remove_stop_words)
df['title_without_stopwords'] = df['title'].apply(remove_stop_words)

df['title_length'] = df['title'].fillna('').apply(lambda x: len(x.split()))
df['text_length'] = df['text'].fillna('').apply(lambda x: len(x.split()))
df['domain_length'] = df['url'].fillna('').apply(lambda x: len(urlparse(x).netloc) if x else 0)
df['platform_risk'] = df['platform'].apply(detect_platform_risk)

# Hebrew
df['text_clean'] = df['text'].apply(remove_hebrew_stop_words)
df['title_clean'] = df['title'].apply(remove_hebrew_stop_words)
