# -*- coding: utf-8 -*-
'''
贝叶斯分类：取后验概率最大的标签

* 先验概率P(Y)，后验概率P(Y|X)
P(Y|X) = P(X|Y)P(Y) / P(X) -- P(X)不考虑

* 贝叶斯决策模型：假设数据X满足高斯分布
  朴素贝叶斯决策模型：假设数据各属性间相互独立，各分布函数相乘得到极大似然部分P(X|Y)
  P(X|Y), P(Y)都能从数据样本中得到; 对于连续型数据假定符合高斯分布

* 拉普拉斯修正：防止条件概率为0最终结果为0 
    P(Yj) = (Dyj + 1)/(D + Ny), P(Xi|Yj) = (Dxiyj + 1)/(Dyj + Nx)

* sklearn实现朴素贝叶斯
    * LabelEncoder().fit_transform
    * naive_bayes.MultinomialNB()
* 

'''

import numpy as np
from sklearn import naive_bayes
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def loaddata():
    X = np.array([[1,'S'],[1,'M'],[1,'M'],[1,'S'],
         [1, 'S'], [2, 'S'], [2, 'M'], [2, 'M'],
         [2, 'L'], [2, 'L'], [3, 'L'], [3, 'M'],
         [3, 'M'], [3, 'L'], [3, 'L']])
    y = np.array([-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1])
    return X, y

X,y = loaddata()
X[:,1] = LabelEncoder().fit_transform(X[:,1])
X = X.astype(int)
# print(X)

model = naive_bayes.MultinomialNB()
model.fit(X,y)
prediction = model.predict(X)
print(prediction)

print(accuracy_score(prediction,y))

