import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from pandasgui import show
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# 1 đọc data
data = pd.read_csv(r'F:\knowleage\python\AI\Machine_Learning\data\Datasets\StudentScore.xls', delimiter=',')

# profile = ProfileReport(data, title="Pandas Profiling Report", explorative=True)
# profile.to_file("StudentScore.html")

# có 3 cột math score, reading score , writting socre chọn cột nào làm target cũng đc
# ví dụ bài này đặt target = writting score 
# có 1 or nhiều feature có correlation >0,7 và <-0,7 so với target nên ta dùng các model regression 

# 2 xem cột nào k quan trọng ta có thể loại bỏ

# 3. phân chia x, y
target = 'writing score'
x = data.drop(target, axis=1)
y = data[target]


# 4 chia bộ train và test
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=42)

# 5 preprocessing cho dữ liệu bị miss trước - bài này k có nên bỏ qua 
# chỉ được xử lý trên bộ train và bộ test là nguyên si
"""
Tác nhân chính là yêu cầu đầu vào của các transformer trong scikit-learn. Chúng luôn đòi hỏi dữ liệu 2 chiều (shape = n_samples × n_features).
df[['cột']] → DataFrame 2 chiều → khớp đúng yêu cầu.
"""

# 6 xử lý dữ liệu trên x_train
# 2 cột số còn lại thì dùng minmaxScaler or StandardScaler là được 
# để k cần tranfrom nhiều lần người ta dùng pipeline - 
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(missing_values = -1, strategy='median')), # trung vị
    ('scaler', StandardScaler())
    # imputer hay scaler chỉ là cái tên đặt ra để ngta biết , còn bên cạnh là các tiền xử lý 
])


"""  dữ liệu kiểu boolean
- cột gender , lunch và test preparation chỉ có 2 giá trị 
- lưu ý với gender phải check xem có đúng nó chỉ có 2 giá trị hay k vì với tính có thể là none hoặc giới tính  t3
- cột lunch cũng chỉ có 2 giá trị 

- có thể dùng chung với ordinal như trong bài - hạn chế dùng với norminal vì nó sinh ra nhiều cột trong khi ordinal chỉ sử lý trong 1 cột
"""
# print(data['gender'].unique())
# print(data['test preparation course'].unique())
# print(data['lunch'].unique())

""" ordinal - có thứ tự
- cột parental level of education - thứ tự các bậc học 
- cột này có dữ liệu : ["bachelor's degree" 'some college' "master's degree" "associate's degree" 'high school' 'some high school'] 
    ta có thể gộp luôn high school và some high school vào vì chúng như nhau
    -> thứ tự tăng dần  :  high school > some college > associate's degree > bachelor's degree > master's degree
"""
# print(data['parental level of education'].unique())

education_values = ['some high school',"high school","some college",  "associate's degree",  "bachelor's degree",  "master's degree"]
gender_values = ['female', 'male']
test_values = ['none' ,'completed']
lunch_values = ['standard', "free/reduced"]

ordinal_tranformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')), # giá trị hay xuất hiện 
    ('encoder',OrdinalEncoder(categories=[education_values, gender_values, test_values, lunch_values])),
])


""" nominal - chỉ đơn giản là chữ cái kí hiệu cho từng chủng tộc - chứ k có thứ tự  
- cột race/ethnicity , - có các giá trị ABCD để phân nhóm
- ngtaa sẽ tránh dùng onehot nhiều tại nó đẻ ra quá nhiều cột
"""

nominal_tranformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(sparse_output=False)),
])


preprocessor = ColumnTransformer(transformers=[
    ("num_feature", num_transformer, ['math score','reading score']),
    ("ordinal_feature", ordinal_tranformer,['parental level of education','gender','test preparation course', 'lunch'] ),
    ("nominal_feature", nominal_tranformer,["race/ethnicity"])
])

reg = Pipeline(steps=[
    ("preprocess", preprocessor),
    # ("model", LinearRegression())
    ("model", RandomForestRegressor())
])


# model.fit
# reg.fit(x_train, y_train)
# y_predict =  reg.predict(x_test)


# ở đây bạn đang dùng pipeline nên bạn buộc phải chỉ định xem tham số nào của cái nào 
# ví dụ ở trên bạn đang muốn chỉnh các tham số như criterion của mô hình Randomforest....
# và bạn đặt tên cho nó là model - ở duowisnbanj phải model__  để chỉ định đó là tham số của mô hình thì chương trình mới hiểu 


# grid_search
pagram = {
    # ví dụ trung num_feature - trong dòng imputer bạn cũng muốn thử các thuộc tính khác của stratgy
    # __ là 2 dấu _  chỉ quan hệ sở hữu
    "preprocess__num_feature__imputer__strategy":["mean", "median"],
    # "preprocess__ordinal_feature__imputer__strategy":["most_frequent", "constant"],
    # "preprocess__nominal_feature__imputer__strategy":['first'],
    "model__n_estimators":[100, 200, 300],
    "model__criterion":["squared_error", "absolute_error", "poisson"],
    "model__max_depth":["None", 2, 5]
}


# vì sao dùng các model phức tạp hơn mà có khi không bằng các model đơn giản 
# do trong bài này , dữ liệu có correlation Lớn với target nên dùng các mô hình tuyến tính là hợp hơn 

grid_search = GridSearchCV(estimator=reg, param_grid=pagram, cv = 5, scoring="r2", verbose=2, n_jobs=-1)
# n_job - dùng số nhân (số luồng trong máy để kiểm soát tiến độ   -1 : là dùng hết luôn )
grid_search.fit(x_train, y_train)
print(grid_search.best_estimator_)
print(grid_search.best_score_)
print(grid_search.best_params_)
y_predict = grid_search.predict(x_test)
print("MSE:  {}".format(mean_squared_error(y_test,y_predict)))
print("MAE:  {}".format(mean_absolute_error(y_test,y_predict)))
print("R2:  {}".format(r2_score(y_test,y_predict))) # càng gần 1 thì càng tốt cứ > 0.8 là ổn

# khi chạy lại có thể ra kq khác - nó tùy theo lúc random dữ liệu ở trên nữa
# 0.928519288074742
# {'model__criterion': 'poisson', 'model__n_estimators': 300, 'preprocess__num_feature__imputer__strategy': 'median'}
# MSE:  19.759449390527845
# MAE:  3.5762871428571428
# R2:  0.9180163580157462


# scoring - mấy chỉ số như mse , sqe, rmqe , ... đều cần càng nhỏ càng tốt vì thế ngta phải đưa về càng to càng tốt 
# như (recall, precition của classification) nên người ta phải thêm dấu âm hay tiền tố "neg" ở trước
# ‘neg_mean_absolute_error’
# ‘neg_mean_squared_error’
# ‘neg_root_mean_squared_error’
# ‘r2’
    

# ngoài grid search ta còn RandomizedSearchCV : grid_search thì nó sẽ thử hết còn RandomizedSearch thì nó sẽ chỉ thử ngẫu nhiên 1 số lượng tổ hợp nhất định mà mình muốn 

# lazyregression 
# reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
# model, predict = reg.fit(x_train, x_test, y_train, y_test)
# print(model)
# print(predict)

