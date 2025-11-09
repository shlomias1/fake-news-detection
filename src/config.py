column_order = [
    'title',
    'text',
    'url',
    'source',
    'category',
    'author',
    'date_published',
    'rating',
    'type',
    'label'
]
high_risk = ['facebook', 'twitter', 'reddit', 'whatsapp', 'telegram', 'tiktok']
low_risk = ['website', 'news site', 'official']
KEYWORDS = ['shocking', 'secret', 'breaking', 'amazing', 'you won’t believe', 'unbelievable']

CATEGORY_MAPPING = {
    'politics': 'Politics',
    'Politics': 'Politics',
    'politicsNews': 'Politics',
    'left-news': 'Left',
    'left': 'Left',
    'right': 'Right',
    'Government News': 'Politics',
    'US_News': 'US',
    'National': 'US',
    'International': 'International',
    'Middle-east': 'International',
    'Editorial': 'Opinion',
    'Finance': 'Finance',
    'Technology': 'Technology',
    'Education': 'Education',
    'Entertainment': 'Entertainment',
    'Sports': 'Sports',
    'Lifestyle': 'Lifestyle',
    'gossip': 'Entertainment',
    'Crime': 'Crime',
    'News': 'News',
    'mainstream': 'News',
    'articles': 'News',
    'stories': 'News',
    'Videos': 'Media',
    'Miscellaneous': 'unknown',
    'unknown': 'unknown',
    'unknow': 'unknown',
    'other': 'unknown',
    '': 'unknown'
}

TYPE_MAPPING = {
    'clickbaits': 'clickbait',
    'Clickbaits': 'clickbait',
    'clickbait': 'clickbait',
    'fake': 'fake',
    'Fake': 'fake',
    'unreliable': 'unreliable',
    'reliable': 'reliable',
    'bias': 'bias',
    'rumor': 'rumor',
    'junksci': 'junk_science',
    'vaccines': 'health',
    'flatearth': 'conspiracy',
    'moonlanding': 'conspiracy',
    'chemtrails': 'conspiracy',
    '911': 'conspiracy',
    'conspiracy': 'conspiracy',
    'hate': 'hate',
    'satire': 'satire',
    'Satire': 'satire',
    'bs': 'fake',
    'all': 'unknown',
    'state': 'unknown',
    '': 'unknown',
    None: 'unknown',
    'unknown': 'unknown'
}

BOILERPLATE_PHRASES = [
    "continue reading", "main story", "advertisement", "email address", "subscribe", 
    "products and services", "special offers", "newsletter", "an error occurred", 
    "try again later", "view new york times newsletters", "receive occasional"
]

HE_STOP = [
    "של","על","עם","אל","מן","מ","ל","ב","כ","ש","ו","ה",
    "הוא","היא","הם","הן","אני","אתה","את","אנחנו","אתם","אתן",
    "זה","זאת","יש","אין","היה","היו","גם","כמו","כל","או","אבל","אם","כי","לא","כן","מה","מי","מתי","איפה","עוד","רק"
]
EN_STOP = [
    "the","a","an","and","or","but","if","because","as","of","at","by","for","with","about","against",
    "between","into","through","during","before","after","to","from","in","out","on","over","under",
    "again","then","once","here","there","when","where","why","how","all","any","both","each","few",
    "more","most","other","some","such","no","nor","not","only","own","same","so","than","too","very",
    "can","will","just","should","now"
]
STOP_ALL = list(set(HE_STOP + EN_STOP))

HEBREW_NIKKUD = r"[\u0591-\u05C7]"  # ניקוד/טעמים להסרה
NON_LETTERS_KEEP_HE = r"[^0-9A-Za-z\u0590-\u05FF\s]"  # משאיר עברית/לטינית/ספרות/רווח
URL_RE   = r"(https?://\S+|www\.\S+)"
EMAIL_RE = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
CLICKBAIT_RE = r"(?i)(לא תאמינו|מזעזע|חובה לראות|צפו|בלעדי|חשיפה|you won't believe|shocking|must see|exclusive|what happened next|watch now)"

LOG_DIR = "fake_news_detection/logs"