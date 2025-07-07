import pandas as pd
import data_io
import utils

def FakeNewsNet_processing():
    gossip_fake = data_io.load_csv(r'data/FakeNewsNet/gossipcop_fake.csv')
    gossip_real = data_io.load_csv(r'data/FakeNewsNet/gossipcop_real.csv')
    political_fake = data_io.load_csv(r'data/FakeNewsNet/politifact_fake.csv')
    political_real = data_io.load_csv(r'data/FakeNewsNet/politifact_real.csv')
    gossip_fake["label"] = 0
    gossip_fake["category"] = "gossip"
    gossip_real["label"] = 1
    gossip_real["category"] = "gossip"
    political_fake["label"] = 0
    political_fake["category"] = "political"
    political_real["label"] = 1
    political_real["category"] = "political"
    FakeNewsNet = pd.concat([gossip_fake, gossip_real, political_fake, political_real], ignore_index=True)
    FakeNewsNet.rename(columns={'news_url': 'url'}, inplace=True)
    FakeNewsNet.drop('tweet_ids', axis=1, inplace=True)
    FakeNewsNet['url'] = FakeNewsNet['url'].astype(str)
    FakeNewsNet['source'] = FakeNewsNet['url'].apply(utils.url_parser)
    FakeNewsNet['author'] = FakeNewsNet['title'].apply(utils.extract_author)
    FakeNewsNet['date_published'] = FakeNewsNet['url'].apply(utils.extract_date_from_url)
    FakeNewsNet['text'] = ''
    FakeNewsNet['sub_category'] = ''
    FakeNewsNet['platform'] = 'website'
    FakeNewsNet['has_image'] = 0
    FakeNewsNet['rating'] = ''
    return utils.reorder_df(FakeNewsNet)

# Index(['id', 'url', 'title', 'text', 'label', 'category', 'type',
#        'author', 'source', 'date_published', 'platform', 'has_image',
#        'meta_rating'],
#       dtype='object')

def fake_true_dataset_processing():
    fake_news = data_io.load_csv(r'data/therealsampat/Fake.csv')
    true_news = data_io.load_csv(r'data/therealsampat/True.csv')
    fake_news["label"] = 0
    true_news["label"] = 1
    fake_true_dataset = pd.concat([fake_news, true_news], ignore_index=True)
    fake_true_dataset.rename(columns={'subject': 'category'}, inplace=True)
    fake_true_dataset.rename(columns={'date': 'date_published'}, inplace=True)
    fake_true_dataset['id'] = pd.Series(fake_true_dataset.index.astype(str)).apply(utils.hash)
    fake_true_dataset['type'] = ''
    fake_true_dataset['url'] = fake_true_dataset['text'].apply(utils.extract_url_from_text)
    fake_true_dataset['author'] = ''
    fake_true_dataset['source'] = ''
    fake_true_dataset['platform'] = 'website'
    fake_true_dataset['has_image'] = 0
    fake_true_dataset['rating'] = ''
    return utils.reorder_df(fake_true_dataset)

# fake_or_real_news Dataset
def fake_or_real_news_processing():
    fake_or_real_news = data_io.load_csv(r'data/fake_or_real_news.csv')
    if 'Unnamed: 0' in fake_or_real_news.columns:
        fake_or_real_news = fake_or_real_news.drop(columns=['Unnamed: 0'])
    fake_or_real_news['id'] = pd.Series(fake_or_real_news.index.astype(str)).apply(utils.hash)
    fake_or_real_news['label'] = fake_or_real_news['label'].str.lower().map({'fake': 0, 'real': 1})
    fake_or_real_news['url'] = fake_or_real_news['text'].apply(utils.extract_url_from_text)
    default_fields = {
        'category': '',
        'type': '',
        'author': '',
        'source': '',
        'date_published': '',
        'platform': 'website',
        'has_image': 0,
        'rating': '',
    }
    for col, default in default_fields.items():
        if col not in fake_or_real_news.columns:
            fake_or_real_news[col] = default
    return utils.reorder_df(fake_or_real_news)

# news_articles Dataset
def news_articales_processing():
    columns = ['author', 'published', 'title', 'text', 'language', 'site_url', 'type', 'label', 'hasImage']
    news_articles = data_io.load_csv(r'data/news_articles.csv')[columns]
    news_articles.rename(columns={'site_url': 'url'}, inplace=True)
    news_articles['id'] = pd.Series(news_articles.index.astype(str)).apply(utils.hash)
    news_articles['platform'] = 'website'
    news_articles['label'] = news_articles['label'].str.lower().map({'fake': 0, 'real': 1})
    news_articles['source'] = news_articles['url'].apply(utils.url_parser)
    news_articles["meta_rating"] = ''
    news_articles.drop('language', axis=1, inplace=True)
    news_articles["category"] = "articles"
    return utils.reorder_df(news_articles)

# liar_data Dataset
test = data_io.load_liar_table(r'data/liar_data/test.tsv')
train = data_io.load_liar_table(r'data/liar_data/train.tsv')
valid = data_io.load_liar_table(r'data/liar_data/valid.tsv')
liar_data = pd.concat([test, train, valid])

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

liar_data.rename(columns={'subject': 'title'}, inplace=True)
liar_data.rename(columns={'statement': 'text'}, inplace=True)

print(liar_data['label'].value_counts())
print(liar_data.columns)