# 机器学习基础

[TOC]

## 梯度下降

* 批量梯度下降BGD

* 随机梯度下降SGD：每次使用一个样本

* 小批量梯度下降MBGD

  

## 评价指标

* 回归：MSE, RMSE, MAE

* 正则化：

  * L2范数正则化-岭回归

  * L1范数正则化-lasso回归 ：解的稀疏性更强

     https://blog.csdn.net/red_stone1/article/details/80755144

  * L1+L2范数-elastic net

* 分类：

  交叉熵loss，accuracy_score
  
  metrics：classification_report(y,pred),accuracy_score,precision_score,recall_score,f1_score，confusion_matrix(y,pred)
  
  

## sklearn



## 1. 分类

### (1) 回归

线性回归

### (2) 分类

#### 	逻辑回归

​    sigmoid函数

####     决策树

* 	ID3：根据熵-信息增益进行特征选择
* 	C4.5：根据熵-信息增益比进行特征选择
* 	CART决策树/Gini基尼指数决策树：二叉树，根据基尼指数进行特征选择

#### 朴素贝叶斯

​	

## 2. 无监督学习

### (1) 聚类

### (2) 降维



## 3. 强化学习

