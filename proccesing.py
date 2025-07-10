import config
import pandas as pd
from urllib.parse import urlparse




# להריץ מודל סיווג כדי להשלים את כל הערכים הלא ידועים בעמודת קטגוריה וטייפ
# להוסיף עמודת שפה
# להוסיף עמודת תרגו לעברית של כותרת ושל תוכן
# key words - בדיקה איזה מילים חוזרות על עצמם בפייק ניוז

df['text_without_stopwords'] = df['text'].apply(remove_stop_words)
df['title_without_stopwords'] = df['title'].apply(remove_stop_words)

df['title_length'] = df['title'].fillna('').apply(lambda x: len(x.split()))
df['text_length'] = df['text'].fillna('').apply(lambda x: len(x.split()))
df['domain_length'] = df['url'].fillna('').apply(lambda x: len(urlparse(x).netloc) if x else 0)
df['platform_risk'] = df['platform'].apply(detect_platform_risk)

# Hebrew
df['text_clean'] = df['text'].apply(remove_hebrew_stop_words)
df['title_clean'] = df['title'].apply(remove_hebrew_stop_words)
