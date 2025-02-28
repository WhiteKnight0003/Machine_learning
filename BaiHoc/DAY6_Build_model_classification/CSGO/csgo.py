import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


data = pd.read_csv(r'F:\AI\Machine_Learning\Datasets\csgo.csv')
print(data)

# x = data.drop(columns=['day', 'month', 'year', 'date', 'team_a_rounds', 'team_b_rounds', 'result', 'map'], axis=1) # xóa tạm map để test hệ số tương quan
x = data.drop(columns=['day', 'month', 'year', 'date', 'team_a_rounds', 'team_b_rounds', 'result'], axis=1)
y = data['result']

num_transformer = Pipeline(steps=[
    "imputer", Im
])
preprocessor  = preprocessor( transformers = [

]
)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)



# print(x.corr().to_string())
# xét hệ số tương quan - nếu hệ số tương quan cao có thể dùng model linear


# sns.histplot(x) # chọn 1 trong các cột số để làm target
# plt.title("csgo infomation")
# plt.savefig(r"F:\AI\Machine_Learning\BaiHoc\DAY6_Build_model_classification\CSGO\csgo.png")
# plt.show()


