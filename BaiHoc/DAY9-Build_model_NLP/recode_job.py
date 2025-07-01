import pandas as pd 
from ydata_profiling import ProfileReport 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import re
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import chi2, SelectKBest, SelectPercentile
# conda install imblearn
from imblearn.over_sampling import RandomOverSampler, SMOTEN


data = pd.read_excel(r'F:\knowleage\python\AI\Machine_Learning\data\Datasets\final_project.ods', engine='odf', dtype=str) # dtype = str - biến all thành str

def filter_location(location):
    res = re.findall("\,\s[A-Z]{2}$", location)
    if len(res) != 0:
        return res[0][2:] # kq ở đây nó trả về 1 cái list nên phải làm v 
    return location

data['location'] = data['location'].apply(filter_location)

# profile = ProfileReport(data, title="Pandas Profiling Report", explorative=True)
# profile.to_file("final_project.html")


# check data_null
# nếu dữ liệu bị khuyết < 5% thì bạn có thể xóa luôn hàng đó đi 
data.dropna(axis=0, inplace=True)
# print(data.isna().sum())


# chia data
target = 'career_level'
x = data.drop(target, axis=1)
y = data[target]


x_train , x_test, y_train , y_test = train_test_split(x, y, train_size= 0.8, random_state = 42, stratify=y)
# train_size or test_size - chỉ đảm bảo tổng số mẫu theo đúng tỉ lệ ví dụ 80 - 20 chứ k đảm bảo tỉ lệ cho từng mẫu - dùng stratify để nó chi tỉ lệ cho cả từng mẫu 

# do inbalance data nên ta cần áp dụng các phương pháp để balance data - chỉ thực hiện cho bộ train
# với việc làm banlance data - chỉ là việc làm dữ liệu đỡ mất cân bằng chứ k phải làm cho nó 1 lô 1 lốc lên 

# randomOverSampler chỉ làm nhân bản mẫu lên chứ k làm đa dạng , muốn đa dậng phải thử smote, adasyn
# ros = RandomOverSampler(random_state=0, sampling_strategy={ # ví dụ chỉ ra là bạn muốn lớp nào được sampleing lên bao nhiêu lần 
#     "director_business_unit_leader" : 500,
#     "specialist" : 500,
#     "managing_director_small_medium_company" : 500,
#     "bereichsleiter" : 1000,
# })



# ValueError: Expected n_neighbors <= n_samples_fit, but n_neighbors = 6, n_samples_fit = 3, n_samples = 3 
# lỗi này là do có những cái sample quá bé k đủ để smoten dùng nên phải thay  đổi chỉ số k_neighbors (mặc định là 5 - ta có thể giảm xuống) 
ros = SMOTEN(random_state=0, k_neighbors=2,sampling_strategy={ # ví dụ chỉ ra là bạn muốn lớp nào được sampleing lên bao nhiêu lần 
    "director_business_unit_leader" : 500,
    "specialist" : 500,
    "managing_director_small_medium_company" : 500,
    "bereichsleiter" : 1000,
})


x_train,y_train = ros.fit_resample(x_train, y_train)


# preprecessing 
# ['title', 'location', 'description', 'function', 'industry','career_level']
# title và description đóng vai trò quan trọng để phân loại và nó là văn bản nên sẽ dùng tf-idf + countvectorizer
# location thì thay vì lấy toàn bộ có thể chỉ lấy 2 kí tự cuối


# dùng onehot encoding để xử lý ở đây - nhưng onehot sinh ra nhiều cột - nên phải làm sao để hạn chế nó  < 50 cột thì được
# word emding - thường dùng trong các bài báo nhưng ở đây location toàn tên riêng - nên k xử lý bằng word emding được

# print(data['function'].unique()) # ít nên có thể dùng onehot
# print(data['industry'].unique()) # nhiều nên dùng tfidfvectorzier


# pipeline chỉ có ý nghĩa với 2 bước - ví dụ bước 1 xử lý trống , bước 2 biến đổi dữ liệu

preprocessor = ColumnTransformer(transformers=[
    ("title_tf",TfidfVectorizer(stop_words='english', ngram_range=(1,1)),'title'),
    ('local', OneHotEncoder(handle_unknown="ignore"), ['location']),
    ('description_tf', TfidfVectorizer(stop_words='english', ngram_range=(1,2), min_df=0.01, max_df=0.95), 'description'),
    ('func', OneHotEncoder(handle_unknown="ignore"), ['function']),
    ('industry_tf', TfidfVectorizer(stop_words='english', ngram_range=(1,1)), 'industry'),
])
# min_df , max_df - token nào xuất hiện ít quá or nhiều quá cũng k tốt nên ngta có cận trên và cận dưới để loại bỏ bớt 




# ValueError: Found unknown categories ['South Carolina', 'West Virginia'] in column 0 during transform
# lỗi này là do khi có quá ít nhãn cho 1 loại - nó nhảy hết vào bộ test - mà lúc train nó k có nên nó lỗi
# phải dùng handle_unknown='ignore' (mặc định là error)
# Thiết lập handle_unknown='ignore' sẽ khiến encoder:
# Không ném lỗi khi gặp category lạ,
# Thay vào đó, gán cho cột one-hot tương ứng toàn bộ giá trị 0 (tức coi đó như “không có category nào đúng”),
# Và vẫn giữ nguyên cấu trúc ma trận đầu ra (số lượng cột one-hot không đổi so với lúc fit).

cls = Pipeline( steps = [
    ('preprocessor', preprocessor),
    # ('feature_selector', SelectKBest(chi2, k=800)), # có thể giảm k thêm - nhưng chỉ giảm đến 1 mức nào đó thôi giảm quá nó sẽ giảm perform của mô hình xuống
    ('feature_selector', SelectPercentile(chi2, percentile=5)), # ở đây là giữ lại k % thuộc tính 
    ('model',RandomForestClassifier())
])

# Hàm chi2 là bài kiểm định Chi-square (χ²) giữa mỗi cột đặc trưng (feature) và biến mục tiêu (target).
# Với mỗi đặc trưng, nó tính giá trị thống kê χ² và p-value, đo lường mức độ phụ thuộc (liên quan) giữa đặc trưng đó và nhãn.
# Tham số k=800 chỉ định giữ lại 800 đặc trưng tốt nhất có giá trị thống kê χ² cao nhất (tức là đặc trưng “liên quan” nhất đến nhãn).

cls.fit(x_train, y_train)
y_predicted = cls.predict(x_test)

print(classification_report(y_test, y_predicted))
