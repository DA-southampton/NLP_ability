##xgboost 训练代码

## 导入相应的包

from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import plot_importance
from matplotlib import pyplot
import pandas as pd
from sklearn import metrics
from sklearn.metrics import classification_report

## 读取对应的数据集
#df = pd.read_csv("./features_not_0.csv",sep = ',')
#df = pd.read_csv("./features_new.txt",sep = ',',header=None)
df = pd.read_csv("./features_new.txt",sep = ',')




"""
省略特征工程
...
...
"""

X = df.iloc[:,1:-1]
Y = df.iloc[:,-1]


seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)


## 模型训练

model = XGBClassifier()
eval_set = [(X_test, y_test)]
model.fit(X_train, y_train)


### 预测
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))



print ('ACC: %.4f' % metrics.accuracy_score(y_test,predictions))
print ('Recall: %.4f' % metrics.recall_score(y_test,predictions))
print ('F1-score: %.4f' %metrics.f1_score(y_test,predictions))
print ('Precesion: %.4f' %metrics.precision_score(y_test,predictions))


