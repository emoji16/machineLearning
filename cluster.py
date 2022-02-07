# -*- coding: utf-8 -*-
'''
聚类算法

* kmeans:手动选取k个质心，对每个点加入最近类，重复计算质心；直到质心不变：需要指定类数
    * 距离计算方法：minkowski距离(n==2时欧氏距离)，cos余弦相似度，皮尔逊相关系数
    * sklearn.cluster KMeans

* 层次聚类
    * 聚合/自下而上聚类：AGNES凝聚层次聚类，类间聚类：需要指定类数
      分裂/自上而下聚类：较少使用
    * sklearn.cluster AgglomerativeClustering

* 密度聚类：不需要指定类数，需要eps和min_samples
    * DBSCAN：r邻域，minpts核心对象，密度直达，密度可达 -- 有可能不属于任何类
      sklearn.cluster DBSCAN

* 高斯混合模型：迭代聚类：需要指定类数
    * 将整个样本看成多个高斯分布，不断提高整个分布的概率值--EM算法，
      指定首先初始化高斯分布数,各高斯分布的参数，然后由结果更新高斯分布参数--迭代算法至收敛
      结果为各类的高斯分布函数
    * from sklearn.mixture import GaussianMixture ,参数：n_components，max_iters
    * trick sklearn.preprocessing import StandardScaler
'''