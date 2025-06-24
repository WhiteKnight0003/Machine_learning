import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from lazypredict.Supervised import LazyClassifier


data = pd.read_csv(r'F:\knowleage\python\AI\Machine_Learning\Datasets\diabetes.csv')

# print(data.describe())
# print(data.head())
# print(data.info())

# đã chạy để tạo file html
# profile = ProfileReport(data, title="Diabetes Dataset Profiling Report", explorative=True)
# profile.to_file("diabetes_profiling_report.html")

target = 'Outcome'
x = data.drop(target, axis = 1)
y = data[target]

# print(x.head())
# print(y.head())

x_train, x_test, y_train,  y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# print(x_train.value_counts())
# print(y_test.value_counts())

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# ở đây khi nhìn vào correlations của các cột với outcome ta đều thấy rằng (hệ số tương quan nhỏ ngoài mức )
# hệ số tương quan lướn khi > 0,7 và < -0,7
# nên ở đây phải dùng các mô hình phi tuyến tính như cây quyết định, rừng ngẫu nhiên, hồi quy logistic
# chý ý : nhiều khi hs tương quan k phản ánh đúng sự liên quan giữa các biến với nhau - do các outlaiers làm hệ số tương quan bị lệch 

# model SVC 
# model = SVC()

# model Logistic Regression
# model = LogisticRegression()

# model Random Forest
# model = RandomForestClassifier(n_estimators=200, random_state=42, criterion='entropy')

# model.fit(x_train, y_train)
# y_predict = model.predict(x_test)

# for i, j in zip(y_test, y_predict):
#     print(f"True: {i}, Predicted: {j}") 

# print(classification_report(y_test, y_predict))
""" SVM
                precision  recall  f1-score   support

          0       0.77      0.83      0.80        99
          1       0.65      0.56      0.60        55

    accuracy                           0.73       154
   macro avg       0.71      0.70      0.70       154
weighted avg       0.73      0.73      0.73       154

- khi data cân bằng thì macro và weighted avg sẽ có ý nghĩa như nhau
- tính macro avg cho từng cột = (0,77 + 0,65 /2) = 0.71 - khi data lệch thì nó k có nhiều ý nghĩa
- tính weighted avg cho từng cột = (0,77 * 99 + 0,65 * 55) / (99 + 55) = 0.73 - khi data lệch thì nó có ý nghĩa hơn
"""

""" Logistic Regression

                precision    recall  f1-score   support

           0       0.81      0.80      0.81        99
           1       0.65      0.67      0.66        55

    accuracy                           0.75       154
   macro avg       0.73      0.74      0.73       154
weighted avg       0.76      0.75      0.75       154
"""


# gridSearch
# params = {
#     'n_estimators': [100, 200, 300],
#     'criterion': ['gini', 'entropy', 'log_loss'],
# }

# model = GridSearchCV(estimator= RandomForestClassifier(random_state=42), param_grid=params, cv=4, verbose=1, scoring='recall')
# """
# param_grid : cần thay đổi cho phù hợp theo các mô hình 

# cv : cross-validation , ở đây cv = 4 tức 
# Chia dữ liệu huấn luyện thành 4 phần bằng nhau
# Ở mỗi vòng lặp, 3 phần được dùng để huấn luyện mô hình, 1 phần còn lại để đánh giá.

# Đánh giá cho từng tổ hợp siêu tham số
# Trong param_grid bạn có 3 giá trị n_estimators × 3 giá trị criterion ⇒ 9 tổ hợp.
# Với 4 fold, tổng số mô hình được huấn luyện là 9 × 4 = 36 (dòng log “Fitting 4 folds for each of 9 candidates, totalling 36 fits” minh họa điều này).
# """

# model.fit(x_train, y_train)
# print(model.best_params_) # in ra params tốt nhất 

# print(model.best_score_) # in ra score tốt nhất - score sẽ tùy vào scoring mà bạn gọi 
# # Tóm lại, khi bạn bỏ qua scoring, best_score_ sẽ là mean accuracy (độ chính xác trung bình k-fold).

# print(model.best_estimator_) # in ra mô hình tốt nhất kèm theo params tốt nhất

# y_predict = model.predict(x_test) # tự động dùng model.bets_estimator_ 
# print(classification_report(y_test, y_predict))


# dùng lazy predict
clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(x_train, x_test, y_train, y_test)
print(models)