# -*- coding: utf-8 -*-
'''
集成学习:群体智慧
    * voting：不同模型，相同数据集，并行结构，对结果投票
        * 局限：各个分类器不完全独立，存在相同偏差
        * 分类：
            硬投票分类器：结果取多
            软投票分类器：概率平均
        * from sklearn.ensemble import VotingClassifier
        voting = VotingClassifier(estimators=[('name',model),...],voting='hard')
        voting.fit()
        voding.predict()

    * bagging：bootstrap aggregating相同模型，不同数据集，并行结构，对结果投票
        * bagging：同一种模型独自放回取样多次训练 （常用）
          pasting：：同一种模型独自不放回取样多次训练
        *随机森林random forest：基分类器是决策树的bagging集成训练
            两种实现：RandomForestClassifier / DecisionTreeClassifier + BaggingClassifier
    * boosting：
        * AdaBoost
        * 提升树BT
        * XGBoost
    * stacking

'''