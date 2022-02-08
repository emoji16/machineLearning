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
  
  

## 1. 分类

### (1) 回归

#### 线性回归

from sklearn.linear_model import ElasticNet

#### 决策回归树：CART

得到分段回归方程

from sklearn.tree import DecisionTreeRegressor

#### DBDT回归

#### XGBoost回归

### (2) 分类

#### 	逻辑回归

sigmoid函数

from sklearn.linear_model import LogisticRegression

####     决策分类树: ID3, C4.5, CART

* 	ID3：根据熵-信息增益进行特征选择
* 	C4.5：根据熵-信息增益比进行特征选择
* 	CART决策树/Gini基尼指数决策树：二叉树，根据基尼指数进行特征选择

from sklearn.tree import DecisionTreeClassifier

#### 贝叶斯决策

from sklearn.naive_bayes import MultinomialNB()

#### 支持向量机

* 线性可分支持向量机： 硬间隔最大化，线性二分类
* 线性支持向量机：软间隔最大化，线性二分类
* 非线性支持向量机：核技巧，非线性分类

from sklearn.svm import LinearSVC

from sklearn.svm import SVC

## 2. 无监督学习

### (1) 聚类

#### Kmeans

from sklearn.cluster import KMeans

#### 层次聚类

AGNES自下而上

from sklearn.cluster import AgglomerativeClustering

#### 密度聚类

from sklearn.cluster import DBSCAN

#### 高斯混合模型

from sklearn.mixture import GaussianMixture

### (2) 降维

#### PCA

* 基于协方差矩阵的特征值分解：X (m*n) * 前k特征向量矩阵转置n*k -- > m*k

* 基于数据矩阵的奇异值分解

from sklearn.decomposition import PCA

## 3. 强化学习



## 4.集成学习

### (1) voting

不同模型，相同数据集，并行结构，对结果投票

* 硬投票分类器：结果取多

* 软投票分类器：概率平均

from sklearn.ensemble import VotingClassifier

### (2) bagging

相同模型，不同数据集，并行结构，对结果投票

* bagging
* pasting

from sklearn.ensemble import BaggingClassifier

#### 随机森林

from sklearn.ensemble import RandomForestClassifier

### (3) boosting

#### AdaBoost分类

多个相同模型串行，加权加和 -- 迭代更新样本权值

 from sklearn.ensemble import AdaBoostClassifier

#### BDTvsGDBT分类回归

BDT：多个树模型串行，加和 -- 每一个模型专注于拟合上一步的残差

GBDT：多个树模型串行，加和 -- 每一个模型专注于拟合当前loss的负梯度

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import GradientBoostingClassifier

#### XGBoost

多个树模型串行，加和 -- 每一个模型取使当前loss最小的树结构和参数取值

pip install xgboost, sklearn没有内置块

### (4) stacking

多个模型并行 + 一层串行

pip install mlxtend, sklearn没有内置块