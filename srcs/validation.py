import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,plot_confusion_matrix
from sklearn.preprocessing import StandardScaler


path_train = 'origin_breast_cancer_data.csv'
data_train = pd.read_csv(path_train)

x= data_train[['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean',
        'concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se',
        'perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave points_se',
        'symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst',
        'smoothness_worst','compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst']]
x=np.array(x).astype('float32')

y_raw = data_train['diagnosis']
y=[]
for each in y_raw:
    if(each=='M'):
        y.append(0)
    else:
        y.append(1)
y=np.array(y).astype('float32')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)

model = LogisticRegression()
model.fit(x_train, y_train)

path_val='breast_cancer_data_357B_100M.csv'
data_val = pd.read_csv(path_val)
x_val_raw= data_val[['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean',
        'concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se',
        'perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave points_se',
        'symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst',
        'smoothness_worst','compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst']]
x_val=np.array(x_val_raw).astype('float32')
y_val_raw = data_val['diagnosis']
y_val=[]
for each in y_val_raw:
    if(each=='M'):
        y_val.append(0)
    else:
        y_val.append(1)
y_val=np.array(y_val).astype('float32')
y_pre_val=model.predict(x_val)

accuracy_score(y_val,y_pre_val)
plot_confusion_matrix(model,x_val,y_val)
print(classification_report(y_val,y_pre_val))

xx = np.arange(0, 1)
yy = xx
plt.plot(xx, yy)
plt.show()
