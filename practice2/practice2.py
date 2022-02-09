# -*- coding: utf-8 -*-
'''
实践 - 有偏数据处理
* 按比例划分train/test：from sklearn.model_selection import StratifiedShuffleSplit
* 平衡比例：且尽可能多保留边界处
    * 过采样/上采样: 随机过采样，SMOTE，borderline SMOTE算法
        * 随机上采样:与旧数据重合 -- from imblearn.over_sampling import RandomOverSampler  
        * SMOTE:在少数样本近邻中点生成新样本 -- from imblearn.over_sampling import SMOTE  
        * Borderline SMOTE：将少数类样本按近邻类型分为noise,danger,safe;只从danger处生成新样本
            Borderline-1 SMOTE 在合成样本时所选近邻是少数类样本(即取少数类样本中点)
            Borderline-2 SMOTE 所有近邻都生成样本、
            在SMOTE参数中增加kind='borderline1'参数即可
        * ADASYN:自适应合成抽样 最后各样本数不完全一样但是总数接近 -- from imblearn.over_sampling import ADASYN
    * 欠采样/下采样: 随机欠采样/原型选择，原型生成，数据清理 
        * 原型选择/随机欠采样：直接在原数据抽取至各类数据数目相同 -- from imblearn.under_sampling import RandomUnderSampler  
        * 原型生成：先聚类，然后取各类中心 -- from imblearn.under_sampling import ClusterCentroids  
        * NearMiss(可看做原型生成)：多个备选启发式规则，选取最有代表性的样本
    * 过采样+欠采样：SMOTE+ENN,SMOTE+Tomek
        * SMOTEENN：-- from imblearn.combine import SMOTEENN
        * SMOTETomek：-- from imblearn.combine import SMOTEENN
    * note：区分负采样 下采样

'''
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
data = pd.read_csv('./data/creditcard.csv')
# print(data['Class'].value_counts())  # 284315 vs 492
X = np.array(data.loc[:,"V1":"V28"])
y = np.array(data['Class'])
sess = StratifiedShuffleSplit(n_splits = 1, test_size = 0.4, random_state = 0)
for train_index, test_index in sess.split(X,y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
