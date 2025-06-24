import pandas as pd
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# import SVM
from sklearn.svm import SVC
# import Logistic Regression
from sklearn.linear_model import LogisticRegression
# import classification_report
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
# import Randomforest
from sklearn.ensemble import RandomForestClassifier
from lazypredict.Supervised import LazyClassifier
#import GridSearchCV
from sklearn.model_selection import GridSearchCV




data = pd.read_csv(r'F:\knowleage\python\AI\Machine_Learning\Datasets\csgo.csv')

# profile = ProfileReport(data, title="CSGO Dataset Profiling Report", explorative=True)
# profile.to_file("csgo_profiling_report.html")

# tỉ lệ correlation thấp => dùng các mô hình phi tuyến 

cols_to_drop = [
    'map','day', 'month', 'year', 'date',
    'wait_time_s', 'match_time_s',
    'team_a_rounds', 'team_b_rounds'
]
data = data.drop(columns=cols_to_drop,axis=1) # or dùng inplace=True

x = data.drop(columns='result', axis=1)
y = data['result']


scaler = MinMaxScaler()

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, random_state=42)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# SVM
# model = SVC()

# logistic regression
# model = LogisticRegression()

# randomforest
# model = RandomForestClassifier()

# model.fit(x_train, y_train)
# y_predict = model.predict(x_test)
# print(classification_report(y_test, y_predict))



# grid search
# param = {
#     'n_estimators': [100, 200, 300],
#     'criterion': ['gini', 'entropy', 'log_loss'],
#     'max_depth': [None, 5, 10]
# }

# model = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param, cv=4, verbose=1, scoring='recall')
# model.fit(x_train, y_train)
# print(model.best_params_)
# print(model.best_score_)
# print(model.best_estimator_)

# y_predict = model.predict(x_test)
# print(classification_report(y_test, y_predict))



# lazy
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
model, predict = clf.fit(x_train, x_test, y_train, y_test)
print(model)


