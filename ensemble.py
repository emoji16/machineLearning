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
        * bagging时不用进行train,test划分，因为会有一定样本不会被抽到所以是test
            bag_cls = BaggingClassifier(SVC(),n_estimator=10,bootstrap=True,max_samples=0.8，oob_score=True)
            print(bag_cls.oob_score_) 使用out_of_bag训练集上的得分
        *随机森林random forest：基分类器是决策树的bagging集成训练
            两种实现：
            * RandomForestClassifier 
                from sklearn.ensemble import RandomForestClassifier
            * DecisionTreeClassifier + BaggingClassifier
    
    * boosting：多个相同模型串行结构(相当于每一步专注处理前一步造成的偏差)
      learning rate：实际应用时对各基模型结果f(x)*alpha设置一个学习率alpha，控制单个模型学习速率
        * AdaBoost分类：多个相同模型串行，数据集相同加权，模型加权加和 -- 迭代更新样本权值
            * 数据权值：每轮训练根据上一轮训练结果改变样本权重(按公式增加/减少样本权重)
            * 模型权值：将各轮训练结果模型加权(由当前模型加权准确率按公式计算)加和组合
            * from sklearn.ensemble import AdaBoostClassifier:para base_estimator,n_estimators,learning_rate
        
        * 提升树BT分类回归：多个分类回归树模型串行，数据集相同，模型加和 -- 每一个模型拟合当前平方loss的负梯度
            * 梯度提升树GBDT vs 提升树BDT：
                提升树：从残差角度依次训练fit(X,y-y')
                梯度提升树：从拟合损失函数的负梯度角度依次训练：损失函数平方时==提升树(特例)
            from sklearn.ensemble import GradientBoostingRegressor:para max_depth,n_estimators,learning_rate
            from sklearn.ensemble import GradientBoostingClassifier:para max_depth,n_estimators,learning_rate
        
        * XGBoost分类回归：多个分类回归树模型串行，数据集相同，模型加和 -- 同样为了min当前平方loss，列出二阶导数泰勒展开+令导数为0。遍历划分选择min_loss的划分方法
            * 损失函数：y+y'损失函数loss + 惩罚项(对叶子节点的个数和叶子值w平方和惩罚)
            * 需要 pip install xgboost
    
    * stacking：多个模型并行+串行结构不同输入，output_1-n = X_input,predict-test_1-n = Y_input_n -> 最后XY结合起来用LR训练
        交叉验证多个模型的output和test_ds的output(类似voting线性组合)的结果组合在一起作为lr的输入
        pip install mlxtend
        from mlxtend.classifier import StackingClassifier

* ps libsvm格式存储稀疏数据: from sklearn.datasets import load_svmlight_file可以读取

'''