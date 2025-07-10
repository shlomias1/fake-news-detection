column_order = [
    'id',
    'title',
    'text',
    'url',
    'source',
    'category',
    'author',
    'date_published',
    'platform',
    'has_image',
    'rating',
    'type',
    'label'
]
high_risk = ['facebook', 'twitter', 'reddit', 'whatsapp', 'telegram', 'tiktok']
low_risk = ['website', 'news site', 'official']
KEYWORDS = ['shocking', 'secret', 'breaking', 'amazing', 'you won’t believe', 'unbelievable']

CATEGORY_MAPPING = {
    'politics': 'politics',
    'left-news': 'politics',
    'government': 'politics',
    'gossip': 'entertainment',
    'worldnews': 'global',
    'health': 'society',
    'covid': 'society',
    'hate': 'extreme',
    'satire': 'humor',
    'fake': 'manipulative'
}

hebrew_stop_words = {
    "של", "על", "את", "עם", "אם", "זה", "הוא", "היא", "היה",
    "אני", "אנחנו", "אתם", "אתן", "הם", "הן", "או", "אבל", "כי",
    "גם", "לא", "כן", "כל", "כמו", "יש", "אין", "אז", "עוד", "מה"
}
