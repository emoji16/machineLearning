# -*- coding: utf-8 -*-
'''
* 逻辑回归
    sigmoid函数
    交叉熵loss

* 逻辑回归实现多分类
    * one-over-rest:训练k个分类器，每个类别对应一个，使用max_score对应的k取类别
        缺陷：训练集(比例)有偏，对结果造成影响
    * one-over-one:训练k*(k-1)/2个分类器，每一对类别对应一个,取出现次数最多的类别
        缺陷:开销大
    * multi-over-multi:对n各类别进行m次划分，训练得到m个分类器
        每一类对应一个m位二进制编码，pick编码最靠近的一类
        理解：上述两者的折中，one-over-one是multi-over-multi的极限情况

* sklearn.linear_model.LogisticRegression
        
* xx,yy = np.meshgrid(np.arange(),np.arange()) 
  z = model.predict(np.c_[xx.ravel(),yy.ravel()]) #np.c_,生成网格坐标
  plt.pcolormesh(xx,yy,z,cmap=plt.cm.Paired)
'''

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 实现plt.show方法
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

def loadData():
    data = np.loadtxt('./data/data1.txt', delimiter=',')
    n_features = data.shape[1]-1
    X = data[:,0:n_features]
    y = data[:,-1].reshape(-1,1)
    return X,y

def plotScatter(X,y):
    pos = np.where(y==1)
    neg = np.where(y==0)
    plt.scatter(X[pos[0],0],X[pos[0],1],marker='x')
    plt.scatter(X[neg[0],0],X[neg[0],1],marker='o')
    plt.show()

def featureNormalization(X):
    x_mean = np.mean(X,axis = 0)
    x_std = np.std(X,axis=0)
    X = (X-x_mean)/x_std 
    return X

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def calLoss(X,y,theta):
    # -ylogy' - (1-y)log(1-y')
    m = X.shape[0]
    loss = - y*np.log(sigmoid(np.dot(X, theta))) - (1-y)*np.log(1-sigmoid(np.dot(X, theta)))
    return np.sum(loss)/m

def gradientDescent(X,y,theta,iterations,alpha):
    m = X.shape[0]
    X = np.hstack((np.ones((m,1)),X))
    for iter in range(iterations):
        theta -= alpha/m * np.dot(X.T,np.subtract(sigmoid(np.dot(X, theta)),y))
        # if iter%10000 == 0:
        #     print('当前进行第{}次迭代, 损失为{}.'.format(iter, calLoss(X,y,theta)))
    return theta

def plotDecisionBoundary(X,y,theta):
    cm_dark = matplotlib.colors.ListedColormap(['b','r'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(X[:,0], X[:,1],c=np.array(y).squeeze(),cmap=cm_dark,s=30)

    x1 = np.arange(min(X[:,0]), max(X[:,0]),0.1)
    x2 = -(theta[1]*x1 + theta[0])/theta[2]
    plt.plot(x1,x2)
    plt.show()

def predict(X,theta):
    X = np.hstack((np.ones((X.shape[0],1)),X))
    res = sigmoid(np.dot(X,theta))
    res[res >= 0.5] = 1
    res[res < 0.5] = 0
    return res

if __name__ == '__main__':
    X,y = loadData()
    # plotScatter(X,y)
    X = featureNormalization(X)
    theta = np.zeros(X.shape[1]+1).reshape(-1,1)
    iterations = 250000
    alpha = 0.008

    theta1 = gradientDescent(X,y,theta,iterations,alpha)
    plotDecisionBoundary(X,y,theta1)

    model = LogisticRegression(C=50, max_iter=2000) # C = 1/lamda
    model.fit(X,y)
    theta2 = np.append(model.intercept_,model.coef_)
    plotDecisionBoundary(X,y,theta2)
    
    print("accuracy_score1:", accuracy_score(y,predict(X,theta)))
    print("accuracy_score2:", accuracy_score(y,model.predict(X)))