import requests
 
# https://github.com/KaiDMML/FakeNewsNet/tree/master/dataset
# https://www.kaggle.com/code/therealsampat/fake-news-detection/input
# https://www.kaggle.com/code/vahidehdashti/detecting-fake-news/input
# https://www.kaggle.com/code/ruchi798/how-do-you-recognize-fake-news/input
# https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/

def get_wayback_snapshot(url):
    api_url = "http://archive.org/wayback/available?url=" + url
    try:
        response = requests.get(api_url)
        data = response.json()
        work_url = data.get("archived_snapshots", {}).get("closest", {}).get("url")
        is_work =  data.get("archived_snapshots", {}).get("closest", {}).get("available") 
        return work_url, is_work
    except:
        return None, False

tst = get_wayback_snapshot("https://en.wikipedia.org/wiki/Michelle_Tanner")
print(tst)