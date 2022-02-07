# -*- coding: utf-8 -*-
'''
支持向量机分类算法：SVM
* 支持向量/支撑向量：支持/支撑超平面的向量向量点

* 分类:
  线性可分支持向量机： 硬间隔最大化，线性二分类
  线性支持向量机：软间隔最大化，线性二分类
  非线性支持向量机：核技巧，非线性分类

* 线性可分支持向量机：优化目标 最近距离最大;label +-1
    * 几何间隔：点到直线距离
      函数间隔
    * 目标函数：带约束的优化 -- 拉格朗日乘子法求极值
        y(i)(wx(i)+b) >= 1 ; min||w||^2
    * 拉格朗日解决带不等式约束的极值条件：KKT条件
        * L = f + alpha *g ; 求g <= 0 + min f
        KKT条件：
            1. L' == 0
            2. alpha >= 0
            3. alpha * g == 0
    * 

* 线性支持向量机：加入松弛变量、容忍度

* 非线性支持向量机：核心思想变换到高维坐标空间，从而线性可分
    * f(x)变换坐标空间--变成线性可分
    * L函数同线性可分支持向量机 ,其中x dot z 替换成K(x,z) = f(x)点乘f(z) 即可
      不用探究f，只找到K核函数，避免高维空间上计算复杂
    * 常用核函数：
        k(x,z) = (x dot z + 1) ^ p
        k(x,z) = exp(-||x-z||^2/(2*方差)) -- rbf，方差为超参数

* 求解方法--拉格朗日乘子法 + SMO算法：求参数alpha1，alpha2->w,b...方法
    每次选择选择alphai，alphaj进行计算。 迭代优化
    alpha !  = 0|C 对应sum(alpha*yi*k(x,xi))+b == 1 为支持向量

* sklearn.svm
    * from sklearn.svm import LinearSVC -- 线性支持向量机,para:C
    * from sklearn.svm import SVC -- 线性支持向量机,para:C,kernel,gamma
        * gamma核函数系数：1/sigma平方，gamma越大，sigma越小，高速分布约高瘦
    * model.n_support_支持向量的个数
      model.support_支持向量在源数据中的索引
      model.support_vectors_支持向量
      model.dual_coef_支持向量的alpha值，拉格朗日乘子法参数
    
    * trick: sklearn集成模型方法 pipeline
        from sklearn.pipeline import make_pipeline
        pca = PCA()
        svc = SVC()
        model = make_pipeline(pca,svc)
    * trick: 此时使用GridSearchCV
    param_grid = {'svc__C':[],'svc__gamma':[]}
    grid = GridSearchCV(model, param_grid)
    grid.fit(X,y)
    print(grid.best_params_)
    model = grid.best_estimator_

'''