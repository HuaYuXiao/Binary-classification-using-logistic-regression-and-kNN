import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,plot_confusion_matrix
from sklearn.preprocessing import StandardScaler

#数据集
path_train = 'origin_breast_cancer_data.csv'
# 使用pandas读入，读取文件中所有数据
data_train = pd.read_csv(path_train)

# 按列分离数据
x= data_train[['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean',
        'concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se',
        'perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave points_se',
        'symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst',
        'smoothness_worst','compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst']]
x=np.array(x).astype('float32')

#读取某列
y_raw = data_train['diagnosis']
y=[]
for each in y_raw:
    if(each=='M'):
        y.append(0)
    else:
        y.append(1)
y=np.array(y).astype('float32')

# 将数据进行拆分，一份用于训练，一份用于测试和验证
# 测试集大小为30%,防止过拟合
# 这里的random_state就是为了保证程序每次运行都分割一样的训练集和测试集。否则，同样的算法模型在不同的训练集和测试集上的效果不一样。
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)


# 逻辑回归模型
model = LogisticRegression()
model.fit(x_train, y_train)  # 训练数据,学习模型参数
y_predict = model.predict(x_test)  # 预测


accuracy_score(y_test,y_predict)
plot_confusion_matrix(model,x_test,y_test)
print(classification_report(y_test,y_predict))


# 绘制真实值和预测值的对比图
xx= np.arange(0, 1)
yy=xx
plt.plot(xx, yy)
plt.title('LogisticRegression')
plt.show()
