# # import requests
# # import pandas as pd

# # # الرابط
# # url = r"https://parttec.onrender.com/part/getAllParts"

# # # طلب البيانات
# # response = requests.get(url)

# # # تحويل الرد إلى JSON
# # data = response.json()

# # # تحويل البيانات إلى DataFrame
# # df = pd.DataFrame(data)

# # # عرض أول 5 صفوف
# # print(df.head())

# # # إذا أردت حفظ البيانات في ملف CSV
# # df.to_csv("Data/parts_data.csv", index=False)
# import requests
# import pandas as pd
# import difflib

# url = "https://parttec.onrender.com/part/getAllParts"

# def fetch_json(url, timeout=15):
#     try:
#         r = requests.get(url, timeout=timeout)
#         r.raise_for_status()
#     except Exception as e:
#         raise RuntimeError(f"خطأ في طلب الـ URL: {e}")
#     try:
#         return r.json()
#     except ValueError:
#         raise RuntimeError(f"الرد ليس JSON صالح. نص الرد (مقتطف):\n{r.text[:500]}")

# def extract_list_from_json(data):
#     # إذا كانت القائمة مباشرة
#     if isinstance(data, list):
#         return data
#     # إذا كان dict: حاول العثور على حقل شائع يحتوي على قائمة
#     if isinstance(data, dict):
#         for key in ("data", "result", "items", "parts", "rows"):
#             if key in data and isinstance(data[key], list):
#                 return data[key]
#         # ابحث عن أول قيمة من القيم التي هي قائمة
#         for v in data.values():
#             if isinstance(v, list):
#                 return v
#         # ربما هو dict من id -> item (جميع القيم dict)
#         if all(isinstance(v, dict) for v in data.values()):
#             return list(data.values())
#     # لم نجد قائمة
#     return None

# def find_best_col(df_cols, candidates):
#     # df_cols: قائمة أعمدة أصلية
#     lc_map = {c.lower(): c for c in df_cols}
#     # تحقق من الأسماء المحتملة بالترتيب
#     for name in candidates:
#         if name.lower() in lc_map:
#             return lc_map[name.lower()]
#     # fuzzy match
#     for name in candidates:
#         matches = difflib.get_close_matches(name.lower(), lc_map.keys(), n=1, cutoff=0.6)
#         if matches:
#             return lc_map[matches[0]]
#     return None

# # قائمة الأسماء المحتملة لكل عمود مطلوب
# candidates_map = {
#     "year": ["year", "manufacture_year", "year_of_make", "model_year", "production_year", "yearOfMake"],
#     "manufacturer": ["manufacturer", "maker", "brand", "company", "manufacture", "manufacturer_name"],
#     "part_name": ["part_name", "partName", "name", "title", "description", "product_name"],
#     "item_id": ["item_id", "id", "itemId", "part_id", "partId", "_id", "uuid"]
# }

# # تنفيذ العملية
# data = fetch_json(url)
# parts_list = extract_list_from_json(data)
# if parts_list is None:
#     raise RuntimeError("لم أتمكن من استخراج قائمة أجزاء من الـ JSON. طالع بنية الـ JSON باليد.")

# # تسطيح (flatten) العناصر المتداخلة
# df_raw = pd.json_normalize(parts_list)

# # حاول إيجاد أعمدة مطابقة
# mapped = {}
# for tgt, cand in candidates_map.items():
#     mapped[tgt] = find_best_col(df_raw.columns.tolist(), cand)

# # أنشئ الـ data_frame المحتوى فقط على الأعمدة المطلوبة (أو None إذا غير موجودة)
# n = len(df_raw)
# data_frame = pd.DataFrame({
#     tgt: (df_raw[mapped[tgt]].copy() if mapped[tgt] is not None else pd.Series([None]*n))
#     for tgt in candidates_map.keys()
# })

# # تحويلات بسيطة
# if "year" in data_frame.columns:
#     data_frame["year"] = pd.to_numeric(data_frame["year"], errors="coerce")

# # تأكيد أنواع وتهيئة item_id كسلسلة نصية (لتفادي مشاكل مع أرقام/None)
# if "item_id" in data_frame.columns:
#     data_frame["item_id"] = data_frame["item_id"].astype("string")

# # حفظ الملف (اختياري)
# data_frame.to_csv("Data/parts_data.csv", index=False)
# data_frame.to_pickle("Data/parts_data.pkl")

# # طباعة ملخص ومعلومات المابّنج
# print("عدد الصفوف:", len(data_frame))
# print("الأعمدة الموجودة الآن في data_frame:", data_frame.columns.tolist())
# print("خريطة الأعمدة (المكتشفة في JSON -> أعمدة DataFrame):")
# for k, v in mapped.items():
#     print(f"  {k}  <-  {v}")
# print("\nأول 5 صفوف:")
# print(data_frame.head())
import pandas as pd
import numpy as np
import random

# -------------------------
# 1. قائمة المستخدمين (user_id جاهزة)
user_ids = [
    "689f84b899fbf5076c5ec985",
    "689f850399fbf5076c5ec989",
    "689f87a099fbf5076c5ec98c",
    "68a02ac4b9fb569a6ae6b733",
    "68a037f0368971a3a299acd8",
    "68a1c4282dcd385eea223e67",
    "68a2008d0e824ad4a74affdf",
    "68a4e27529c853d3c50a06b9",
    "68a989b7d3101920ef9dcdc5",
    "68a989f2d3101920ef9dcdc9",
    "68b70934570891ebac502084",
    "68b71602570891ebac502242",
    "68b873c9c943d0109489951a",
    "68b8bf970f9799245af77682",
    "68b94594f7f0ff7d31298ae8",
    "68b98efec02accb223b32cbf"
]

users_df = pd.DataFrame({
    "user_id": user_ids,
    "name": [f"user_{i}" for i in range(len(user_ids))]
})

# -------------------------
# 2. تحميل items من ملف CSV
# (ملف parts_data.csv فيه أعمدة: item_id, part_name, manufacturer, year)
items_df = pd.read_csv("Data/parts_data.csv")

# -------------------------
# 3. توليد بيانات التوصية
rating_size = 100   # عدد سجلات التقييم
order_size = 100    # عدد الطلبات

ratings_df = pd.DataFrame({
    "user_id": np.random.choice(users_df["user_id"], size=rating_size),
    "item_id": np.random.choice(items_df["item_id"], size=rating_size),
    "rating": np.random.randint(1, 6, size=rating_size)   # تقييم من 1 إلى 5
})

orders_df = pd.DataFrame({
    "user_id": np.random.choice(users_df["user_id"], size=order_size),
    "item_id": np.random.choice(items_df["item_id"], size=order_size)
})

# -------------------------
# 4. حفظ النتائج (اختياري)
users_df.to_csv("Data/users.csv", index=False)
ratings_df.to_csv("Data/ratings.csv", index=False)
orders_df.to_csv("Data/orders.csv", index=False)

print("Users:", users_df.shape)
print("Items:", items_df.shape)
print("Ratings:", ratings_df.shape)
print("Orders:", orders_df.shape)
