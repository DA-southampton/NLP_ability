## xgboost获得特征重要程度并画图

from numpy import loadtxt
from xgboost import XGBClassifier

# 读取数据集
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")

# 拆分数据为特征和目标

X = dataset[:,0:8]
y = dataset[:,8]

# 模型拟合数据
model = XGBClassifier()
model.fit(X, y)

# 打印模型特征重要程度并画图显示
print(model.feature_importances_)
print(model.feature_importances_)
plot_importance(model)
pyplot.show()