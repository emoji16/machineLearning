# -*- coding: utf-8 -*-
'''
pca:降维--特征选择/图像压缩，k == 1 时相当于往直线上做投影
    * 法1：基于协方差矩阵的特征值分解 -- para：降低后的维度
        s1 - 计算归一化特征协方差矩阵X.T*X; 
        s2 - 计算得到r个特征值和r*n特征向量U np.linalg.eig(X);
        s3 - 按特征值降序选择k个特征向量，k*n -> n*k;
        s4 - (X) m * n * n * k -- m * k降维过程;
        s5' - X*U = Z，近似复原X'= Z*UT -->直线上的投影
        P'XP = E
        理解：特征值对应的特征向量就是理想中想取得正确的坐标轴，而特征值就等于数据在旋转之后的坐标上对应维度上的方差。
    * 法2：基于数据矩阵的奇异值分解
        s1 - 计算归一化特征协方差矩阵X; 
        s2 - u,s,vt = np.linalg.svd(X,full_matrices = 0) # m*n,n*n,n*n,默认n=features_num
        s3 - 压缩n->r : m*r,r*r,r*n-vt 
        s4 - 降维：X = X*vt.T -> m*r
        
法1:
from sklearn.decomposition import PCA
model = PCA(n_components = 1)
Z = model.fit_transform(X)
model.explained_variance_ratio_
model.explained_variance_
X_new = model.inverse_transform(Z)
'''