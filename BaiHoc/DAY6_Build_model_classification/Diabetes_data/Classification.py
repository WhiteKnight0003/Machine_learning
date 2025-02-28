import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import  make_scorer, precision_score

data = pd.read_csv(r'F:\AI\Machine_Learning\Datasets\diabetes_data.csv')
# print(data.isnull().sum())

target = "DiabeticClass"
# tách dữ liệu
x = data.drop(columns=[target], axis=1)
y = data[target]

# chia bộ test và train
x_train, x_test, y_train, y_test = train_test_split(x, y , train_size=0.8, random_state=42, stratify=y)

# về giới tính nên kiểm tra chắc
# print(x["Gender"].unique())
# =>  trong bài này age là cột số còn đâu các cột còn lại đều là boolean


# xử lý thiếu và xử lý dạng StandardScaler
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
     ("scaler", MinMaxScaler()),
    # dòng 1 là dòng điền dữ liệu trống
    # dòng 2 là tham số - để tiền xử lý dữ liệu
])


# print(data.columns)

gender_val = x_train["Gender"].unique()
excess_val = x_train["ExcessUrination"].unique()
polydipsia_val = x_train["Polydipsia"].unique()
weight_val = x_train["WeightLossSudden"].unique()
fat_val = x_train["Fatigue"].unique()
polyphagia_val = x_train["Polyphagia"].unique()
genital_val = x_train["GenitalThrush"].unique()
blurr_val = x_train["BlurredVision"].unique()
itching_val = x_train["Itching"].unique()
irri_val = x_train["Irritability"].unique()
delay_val = x_train["DelayHealing"].unique()
partial_val = x_train["PartialPsoriasis"].unique()
muscle_val = x_train["MuscleStiffness"].unique()
alopecia_val = x_train["Alopecia"].unique()
obesiry_val = x_train["Obesity"].unique()



# dùng ordinal cho dạng boolean
ord_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
    ("encoder", OrdinalEncoder(categories= [ gender_val, excess_val, polydipsia_val, weight_val, fat_val, polyphagia_val, genital_val, blurr_val, itching_val, irri_val, delay_val, partial_val, muscle_val, alopecia_val, obesiry_val ]))
])


preprocessor = ColumnTransformer(transformers = [
    ("num_feature", num_transformer, ["Age"]),
    ("odr_feature", ord_transformer, ['Gender', 'ExcessUrination', 'Polydipsia', 'WeightLossSudden',
       'Fatigue', 'Polyphagia', 'GenitalThrush', 'BlurredVision', 'Itching',
       'Irritability', 'DelayHealing', 'PartialPsoriasis', 'MuscleStiffness',
       'Alopecia', 'Obesity']),
])

reg = Pipeline(steps=[
    ("proprecessor",preprocessor),
    ("classifier", RandomForestClassifier())
])

parameters = {
    "classifier__criterion" : ['gini', 'entropy', 'log_loss'],
    "classifier__max_features" : ['sqrt', 'log2', None],
    "classifier__min_samples_split" : [2, 5, 10],
    "classifier__n_estimators" : [50,100],
    "classifier__max_depth" : [ None, 5, 10 ],
}

# Nếu bạn muốn sử dụng scoring='precision' trong GridSearchCV (hoặc một phương pháp tương tự) và nhãn của bạn là ['Negative', 'Positive'] (chuỗi ký tự thay vì số),
# bạn cần xử lý tương tự như với f1_score, vì lỗi pos_label=1 is not a valid label cũng sẽ xảy ra với precision

precision_scorer  = make_scorer(precision_score, pos_label="Positive" )
cls = GridSearchCV(reg , param_grid=parameters, n_jobs=-1,  cv=6, verbose=1, scoring=precision_scorer) # n_jobs=-1 lấy hết nhân
cls.fit(x_train, y_train)
print(cls.best_score_)
print(cls.best_params_) # đưa ra tham số tốt nhất cho model
y_pred = cls.predict(x_test)

# for i,j in zip(y_test, y_pred):
#     print('i = {}, j= {}'.format(i, j))

print(classification_report(y_test, y_pred))