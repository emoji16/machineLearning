# -*- coding: utf-8 -*-
'''
gradient descent实现一元线性回归 - np实现
    * MSE
    * ridge regression
    * lasso regression

最小二乘法：一步到位直接求导数为零情况 - np实现
    * 限制：计算量大且要求有逆

sklearn - model,.fit,.intercept_,.coef_,.predict
    * 提供训练集验证集测试集及处理
        sklearn.datasets import load_
        sklearn.model_selection.train_test_split
    * 提供模型 sklearn.linear_model
    * 模型选择 sklearn.model_selection.GridSearchCV
        sklearn.model_selection import GridSearchCV确定超参数取值 + cv交叉验证
        gSearch = GridSearchCV(estimator=ridge_model,param_grid=para_list,cv=5，scoring='neg_mean_squared_error')
        gSearch.fit(X_train,y_train)
        查看gSearch.best_params_, gsearch.best_score_
        para_list为字典
    * 模型评价 sklearn.metrics：mean_squared_error...
    * 模型保存部署 sklearn.externals.joblib: dump load
        joblib.dump(model,model_path)
        model = joblib.load(model_path)


'''

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 实现plt.show方法
import matplotlib.pyplot as plt
from sklearn import linear_model

def loadData():
    data = np.loadtxt('./data/data1.txt', delimiter=',')
    n_features = data.shape[1]-1
    X = data[:,0:n_features]
    y = data[:,-1].reshape(-1,1)
    return X,y

def featureNormalization(X):
    x_mean = np.mean(X,axis = 0)
    x_std = np.std(X,axis=0)
    X = (X-x_mean)/x_std 
    return X

def gradientDescent(X,y,theta,iterations,alpha):
    c = np.ones(X.shape[0]).transpose()
    X = np.insert(X,0,values=c,axis=1)
    m = X.shape[0]
    losses = []

    # python中矩阵乘法: multiply, * 点乘; dot 乘积
    for iter in range(iterations):
        # python中矩阵加法: np.add
        theta  = np.add(theta, alpha/m *np.dot(X.transpose(),y-np.dot(X,theta)))
        losses.append(np.sum(np.power(np.dot(X,theta) - y,2))/(2*X.shape[0]))
    return theta, losses

def ridgeRegression(X,y,theta,iterations,alpha,lamda):
    c = np.ones(X.shape[0]).transpose()
    X = np.insert(X,0,values=c,axis=1)
    m = X.shape[0]
    losses = []

    # python中矩阵乘法: multiply, * 点乘; dot 乘积
    for iter in range(iterations):
        # python中矩阵加法: np.add
        theta = np.add(theta, alpha/m *np.dot(X.transpose(),y-np.dot(X,theta)) - 2*alpha*lamda*theta) 
        losses.append(np.sum(np.power(np.dot(X,theta) - y,2))/(2*X.shape[0]))
    return theta, losses

def lassoRegression(X,y,theta,iterations,alpha,lamda):
    c = np.ones(X.shape[0]).transpose()
    X = np.insert(X,0,values=c,axis=1)
    m = X.shape[0]
    losses = []
    # python中矩阵乘法: multiply, * 点乘; dot 乘积
    for iter in range(iterations):
        # python中矩阵加法: np.add
        tmp = [1 if theta_x >= 0 else -1 for theta_x in theta]
        tmp = np.asarray(tmp).reshape(-1,1)
        theta = np.add(theta, alpha/m *np.dot(X.transpose(),y-np.dot(X,theta)) - 2*alpha*lamda*tmp) 
        losses.append(np.sum(np.power(np.dot(X,theta) - y,2))/(2*X.shape[0]))
    return theta, losses

def leastSquaredRegression(X,y):
    c = np.ones(X.shape[0]).transpose()
    X = np.insert(X,0,values=c,axis=1)
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return theta

def predict(X, theta):
    X = featureNormalization(X)
    c = np.ones(X.shape[0]).transpose()
    X = np.insert(X,0,values=c,axis=1)
    return np.dot(X,theta)

if __name__ == '__main__':
    X,y = loadData()
    X = featureNormalization(X)
    theta = np.zeros(X.shape[1]+1).reshape(-1,1)
    iterations = 400
    alpha = 0.01
    lamda = 0.02

    # theta1,losses1 = gradientDescent(X,y,theta,iterations,alpha)
    # theta2,losses2 = ridgeRegression(X,y,theta,iterations,alpha,lamda)
    # theta3,losses3 = lassoRegression(X,y,theta,iterations,alpha,lamda)
    # print(theta3)
    # theta4 = leastSquaredRegression(X,y)

    # model1 = linear_model.LinearRegression()
    # model1.fit(X,y)
    # print(model1.intercept_, model1.coef_)
    # model_y1 = model1.predict(X)

    # model2 = linear_model.Ridge(alpha = lamda)
    # model2.fit(X,y)
    # print(model2.intercept_, model2.coef_)
    # model_y2 = model2.predict(X)

    # model3 = linear_model.Lasso(alpha = lamda)
    # model3.fit(X,y)
    # print(model3.intercept_, model3.coef_)
    # model_y3 = model3.predict(X)

    model4 = linear_model.ElasticNet(alpha = lamda)
    model4.fit(X,y)
    print(model4.intercept_, model4.coef_)
    model_y4 = model4.predict(X)

    plt.scatter(X,y)
    test_x = np.linspace(-2,5,25).reshape(-1,1)
    # test_y1 = predict(test_x,theta1) 
    # test_y2 = predict(test_x,theta2) 
    # test_y3 = predict(test_x,theta3) 
    # test_y4 = predict(test_x,theta4) 
    # plt.plot(test_x,test_y1,color='orange')
    # plt.plot(test_x,test_y2,color='blue')
    # plt.plot(test_x,test_y4,color='green')
    # plt.plot(test_x,test_y4,color='pink')
    plt.scatter(X,model_y4,color='red')
    plt.show()

    # iter = np.linspace(1,len(losses1),len(losses1))
    # plt.plot(iter,losses1,color='orange')
    # plt.plot(iter,losses2,color='red')
    # plt.plot(iter,losses3,color='green')
    # plt.show()
