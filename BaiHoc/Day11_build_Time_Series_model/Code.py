# build Times series Forecasting model
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# create time series data
def create_ts_data(data, window_size=5):
    i = 1
    while (i < window_size):
        data['co2_{}'.format(i)] = data['co2'].shift(-i)  # if i , you create wrong direction column
        i += 1  # shift 1 unit
    data['target'] = data['co2'].shift(-i)  # create target column
    data = data.dropna(axis=0)  # drop missing value in end dataframe , beacause we is in end data and we continous
    # you can end continous data but you can not missing value at mid data
    return data


# Load data
data = pd.read_csv(r'F:\AI\Machine_Learning\Datasets\Time-series-datasets\co2.csv')

# Check data
# print(data)

# Check data type
# print(data.dtypes)

# with time column , it is better to convert to datetime
# Convert date to datetime
data['time'] = pd.to_datetime(data['time'])

# draw plot
# fig : large frame
# ax : small frame

# # Check missing value
# print(data.isnull().sum())

# with classification and regression problem , if data missing small 5 percent , we can drop it
# BUT time series forecasting , we can not drop it because data in problem is Countinous data . If we drop miss data , data is broken

# How Fill missing value :
# interpolate (nội suy - dùng dữ liệu ở trước sau để điền vào giữa): fill missing value by mean of 2 value before and after missing value
# extrapolate (ngoại suy - ngược nội suy): fill missing value by mean of 2 value before and after missing value

# Fill missing value by interpolate
data['co2'] = data['co2'].interpolate()

# Check missing value
# print(data.isnull().sum())

# init frame
# fig, ax = plt.subplots()
# # Point axis x, y
# ax.plot(data['time'], data['co2'])
# ax.set_xlabel('Year')
# ax.set_ylabel('CO2')
# # show plot
# plt.show()
#


# handel model
# tạo ra 1 cột mới , mỗi lần dịch 1 đơn vị , tạo đủ 5 bản sao và cái cột 6 sẽ là đầu ra
# ceate new column , each time shift 1 unit , create 5 copy and 6th column is output

#     1
#   1 2
# 1 2 3 ....
# 2 3 4
# 3 4
# 4

# call function
data = create_ts_data(data)

# divide data
x = data.drop(columns=['time', 'target'], axis=1)
y = data['target']

# divide train and test data
# we can not use train_test_split because it is random split

train_ratio = 0.8
num_sample = len(x)

res = int(num_sample * train_ratio)
x_train = x[:res]  # only [0 -> res-1 ] 
y_train = y[:res]

x_test = x[res:]
y_test = y[res:]
