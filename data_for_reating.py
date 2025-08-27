import pandas as pd 

link = r'C:\Users\USER\.cache\kagglehub\datasets\anushabellam\amazon-reviews-dataset\versions\1\amazon_grocery_Data.csv'
data = pd.read_csv(link)

data.columns


data[['customer_id','review_id','star_rating']]
data['star_rating'].value_counts()


import requests
import gzip
import json
import pandas as pd

# مثال: تحميل بيانات Amazon 2023 – Automotive (رابط وهمي، يرجى استبداله بالرابط الحقيقي)
url = "https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_reviews_us/Automotive_v1_00.json.gz"

# تحمّل الملف
r = requests.get(url, stream=True)
r.raise_for_status()

# كتابة الملف محليا
fname = "Automotive_v1_00.json.gz"
with open(fname, "wb") as f:
    for chunk in r.iter_content(chunk_size=1024):
        f.write(chunk)

# قراءة أول 50,000 سجل فقط (لتجربة خفيفة)
records = []
with gzip.open(fname, 'rt', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 50000:
            break
        records.append(json.loads(line))

df = pd.DataFrame(records)
print("First columns:", df.columns)
print(df.head())

# تجهيز dataframes لتناسب نموذجك
ratings_df = df[["reviewerID", "asin", "overall"]].dropna()
ratings_df.columns = ["user_id", "item_id", "rating"]

orders_df = ratings_df[["user_id", "item_id"]]

# حفظ الملفات
ratings_df.to_csv("amazon_automotive_ratings.csv", index=False)
orders_df.to_csv("amazon_automotive_orders.csv", index=False)

print("تم حفظ البيانات: amazon_automotive_ratings.csv و amazon_automotive_orders.csv")


import json

file = # e.g., "All_Beauty.jsonl", downloaded from the `review` link above
file = 'All_Beauty.jsonl'
with open(file, 'r') as fp:
    for line in fp:
        print(json.loads(line.strip()))


from datasets import load_dataset
import pandas as pd

# تحميل بيانات MerRec - روابط Parquet متوفرة من Hugging Face
dataset_url = "https://huggingface.co/datasets/mercari-us/merrec/resolve/main/data/interactions/part-00000.parquet"

# تحميل أول 50,000 صف فقط لتقليل الحجم
dataset = load_dataset("parquet", data_files=dataset_url, split="train[:50000]")


from datasets import load_dataset

ds = load_dataset("mercari-us/merrec",data_files='https://huggingface.co/datasets/mercari-us/merrec/resolve/main/data/interactions/202305/')

shard_url = (
    "https://huggingface.co/datasets/mercari-us/merrec/resolve/main/data/interactions/202305/"
    "part-00000-*.parquet"
)
