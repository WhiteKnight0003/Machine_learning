import pandas as pd
from pyexpat import features
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, chi2, SelectPercentile
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import RandomOverSampler, SMOTEN
from sklearn.utils.fixes import percentile
from statsmodels.graphics.tukeyplot import results

#  dùng regex để lấy 2 ký tự cuối - nếu 2 kí tự cuối là mã bang thì lấy còn k thì sẽ lấy tên thành phố
def filter_location(location):
    result = re.findall("\,\s[A-Z]{2}$", location) # tìm ra tất cả các cụm hợp lệ
    if len(result) != 0:
        return result[0][2:] # nếu mà có nhiều cụm giống nhau - chỉ lấy 1 cái
    else:
        return location


data = pd.read_excel(r"F:\AI\Machine_Learning\Datasets\final_project.ods", engine="odf", dtype=str) # mở file và biến all data thành string đ k bị lỗi
# print(data["career_level"].value_counts())

# xóa cái hàng có dữ liệu bị nan
data = data.dropna(axis=0)

# xét lại cột
data["location"] = data["location"].apply(filter_location) # sửa lại cột location
#

print(len(data['function'].unique())) #  # bạn có thể xem có nên dùng OnehotEncoder hay k bằng cách test qua số cột giá trị
# nẾU nó nhỉ có thể dùng OnehotEncder

#  chia data
target = "career_level"
x = data.drop(target, axis=1)
y = data[target]


# tạo bộ bộ test và bộ train
# stratify = y : - bình thường khi bạn chia bộ train và test  nó chỉ đảm bảo tất cả tham số của 2 bộ tí lệ với nhau
#    => khi bạn setting cho stratify = y : nó sẽ chia đều dữ liệu của all column theo đúng tỉ lệ cho cả 2 bộ train và test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100, stratify=y)


# trong phần balance data chúng ta có các cách xử lý data
# ở đây ta dùng overSamling để nhân bản thêm dữ liệu ra
# ở đây nếu ta không xử lý gì nó sẽ rất nguy hiểm vì khi ta nhân bản dữ liệu rồi mới chia nhiều khi dữ liệu sẽ tồn tại ở cả bên bộ train và test  (vi phạm khi 1 sample chỉ có thể nằm trong bộ train or bộ validation or bộ test)
# ros = RandomOverSampler(random_state=0)
# x,y = ros.fit_resample(x, y)
# giải pháp chỉ chia cho bộ train

# nếu chỉ dùng oversampling đơn giản thì sẽ không đa dạng thay vào đó ta sẽ dùng các kỹ thuật con như SMOTE , ADASYN
# lưu ý SMOTE chỉ làm việc được với dữ liệu dạng số vì thế ngta đã phát minh ra SMOTENC để có thể xử lý data dạng chuỗi
# ros = RandomOverSampler(random_state=0, sampling_strategy={

ros = SMOTEN(random_state=0,k_neighbors=2, sampling_strategy={
    # chỉ định giá trị mà bạn muốn oversampling trong column y_train và số lượng muốn oversampling
    "director_business_unit_leader" : 500,
    "specialist" : 500,
    "managing_director_small_medium_company" : 500,
    "bereichsleiter" : 1000,
})

# print(y_train.value_counts())
# print("_________________")
x_train,y_train = ros.fit_resample(x_train, y_train) # việc nhân bản ở đây chỉ để data đỡ mất cân bằng hơn thôi
# print(y_train.value_counts())


# print(y_train.value_counts())
# # x_train, y_train = ros.fit_resample(x_train, y_train)
# print("-----------------")
# print(y_train.value_counts())



# tiền xử lý cho từng cột

# nên dùng Tfidf cho những cột dài
# vectorizer = TfidfVectorizer(stop_words='english')
# result = vectorizer.fit_transform(x_train['title'])
# print(vectorizer.vocabulary_) # 1 dict { từ : vị trí xuất hiện của nó }
# print(len(vectorizer.vocabulary_))
# print(result.shape) # số hàng và số từ trong vocabulary


# những cột ngắn vẫn dùng được Tfidf  nhưng nên thử những cách khác như OnehotEncoder
# đặc biệt đối với các cột nhiều tên riêng k nên dùng Tfidf vì nó k có nhiều ý nghĩa và sẽ bị loại bỏ nhiều nên cần dùng những cách khác
# encoder = OneHotEncoder()
# result = encoder.fit_transform(x_train[['location']])
# print(result.shape) # (6458, 959) - ban đầu bạn không dùng Onehot ở đây được vì nó tạo thêm ra tận 959 cột thêm mới ( < 50 còn có thể chấp nhạn)
# # print(data.isna().sum())

# vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2)) # thay vì lấy 1 từ t có thể lấy 2
# result = vectorizer.fit_transform(x_train["description"])
# print(vectorizer.vocabulary_)
# print(len(vectorizer.vocabulary_))
# print(result.shape)
# mặc định n_ram = (1,1) - kq : - hay  Uni -  (6458, 66674)
# n_gram = (1,2) kq : - hay Uni + bi  - (6458, 846809) - giảm khả năng từ bị trùng lặp

# # ros = SMOTEN(random_state=0, k_neighbors=2, sampling_strategy={
# #     "director_business_unit_leader": 500,
# #     "specialist": 500,
# #     "managing_director_small_medium_company": 500,
# #     "bereichsleiter": 1000
# # })


# min_df giúp loại bỏ các từ/ngram quá hiếm gặp (xuất hiện quá ít → có thể gây nhiễu, ít giá trị phân biệt).
# Ví dụ: min_df=0.01 → chỉ giữ lại các từ/ngram xuất hiện ít nhất trong 1% tổng số văn bản

# max_df giúp loại bỏ các từ/ngram quá phổ biến (hầu như xuất hiện ở khắp mọi nơi → không giúp phân loại).
# Ví dụ: max_df=0.95 → loại bỏ các từ/ngram xuất hiện trong trên 95% tổng số văn bản (tức là những từ quá phổ biến, có thể không mang nhiều thông tin).

preprcessor = ColumnTransformer(transformers=[
    # (Tên transformer , Tiền xử lý , tên cột)
    ('title_ft', TfidfVectorizer(stop_words='english', ngram_range=(1, 1)), 'title'),
    ('location_ft', OneHotEncoder(handle_unknown='ignore'), ['location']), # onehot phải để cột trong []

    # cột này chạy quá lâu phải điều chỉnh lại - để giảm bớt số feature đi - nếu k feature được sinh ra từ ngram_range(1,2) là rất nhiều - làm chậm đến việc chạy model
    # ('description_ft', TfidfVectorizer(stop_words='english', ngram_range=(1, 2)), 'description'),  # ngram_range(1,2) ở đây Lấy cả từ đơn lẻ và cặp từ liên tiếp để nắm bắt thêm ngữ cảnh.
    ('description_ft', TfidfVectorizer(stop_words='english', ngram_range=(1, 2),min_df = 0.01, max_df = 0.95), 'description'),

    ('function_ft', OneHotEncoder(handle_unknown='ignore'), ['function']),
    ('industry_ft', TfidfVectorizer(stop_words='english', ngram_range=(1, 1)), 'industry'), # Chỉ lấy từ đơn lẻ, có thể vì ngành nghề thường là danh từ riêng và không cần kết hợp nhiều từ.
])

cls = Pipeline([
    ("preprocessor", preprcessor),
    # ('feature_selector', SelectKBest(chi2, k=800)), # chi2 ở đây là 1 trong tiêu chí đánh giá của class feature_selector, nếu muốn lấy % thì dùng percentile = n - n ở đây sẽ là số %
    ('feature_selector', SelectPercentile(chi2, percentile = 5)),

    # SelectKBest(chi2, k=800) sẽ tính điểm thống kê chi-squared cho mỗi đặc trưng so với nhãn (target) và chọn ra 800 đặc trưng có điểm số cao nhất.
    ('model', RandomForestClassifier()),
])

# check số feature
# result = cls.fit_transform(x_train)
# print(result.shape)


# trong quá trình chạy sẽ có lỗi như này ValueError: Found unknown categories ['Edmonton', 'Mississippi', 'New Hampshire', 'New Mexico'] in column 0 during ...
# lỗi này là ở cột location - khi bạn apply các giá trị trên quá thiểu số và nó bị lọt hết vào bộ test (bộ train k hề có nên nó sinh ra lỗi là dữ liệu k có ở bộ train)
# cách sửa : thêm thuộc tính handle_unknown : ignore - Điều này giúp bỏ qua các danh mục không xuất hiện trong tập huấn luyện thay vì gây lỗi.


# kq khi chạy cho thấy class nào có nhiều sample thì perfom càng tốt và ngược lại


params = {
    'model__criterion': ['gini', 'entropy', 'log_loss'],
    # 'model_n_estimators': [100, 200, 300],
    'feature_selector__percentile': [1, 5, 10],
}

grid_search = GridSearchCV(estimator=cls,param_grid=params,cv=4,scoring='recall_weighted', verbose=1) # đối với scoring : các thông số nhưu f1, recall , precision chỉ có thể xử lý dữ liệu dạng nhị phân nên buộc phải dùng khác 1 tí
grid_search.fit(x_train, y_train)
y_predict = grid_search.predict(x_test)
print(classification_report(y_test, y_predict))




# về mặt bản chất - khi giá trị feature quá nhỏ và bạn muốn gộp chúng - bạn phải chắc chắn chúng có sự tương đồng
# còn k thì bạn k nên gộp vì nó sẽ làm mất đi sự phân cấp và khi gộp thì class bạn đặt tên cũng sẽ k biết nó đại diện cho đối tượng nào