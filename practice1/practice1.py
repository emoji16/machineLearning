# -*- coding: utf-8 -*-
'''
实践 - 数据预处理与特工程部分
1 - categorical_encoder
2 - 管道式数值/离散型数据处理
'''
import pandas as pd
import matplotlib.pyplot as plt
from categorical_encoder import CategoricalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

bank_rec = pd.read_csv('./data/bank-full.csv',sep=';')
# print(bank_rec.head())

# print(bank_rec.describe())
# print(bank_rec.describe(include=['0']))  # 查看离散型变量
# print(bank_rec.info())
# print(bank_rec['y'].value_counts()) #note1：有偏数据no     39922,yes     5289
# bank_rec.hist(bins=25,figsize=(14,10))
# plt.show()

# s1 - CategoricalEncoder : categorical_encoder.py
# bank_rec = CategoricalEncoder().fit_transform(bank_rec[['job']])
# print(bank_rec.toarray())

# # s2 - pipeline

numeric_cols = bank_rec.select_dtypes(include='number').columns.values.tolist()
categorical_cols = bank_rec.select_dtypes(include='object').columns.values.tolist()
categorical_cols.remove('y')

#DataFrameSelector类的作用是从DataFrame中选取特定的列，以便后续pipeline的便捷性。
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]

numerical_pipeline = Pipeline([
    ("select_numeric", DataFrameSelector(numeric_cols)),
    ("std_scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("select_cat", DataFrameSelector(categorical_cols)),
    ("cat_encoder", CategoricalEncoder(encoding='onehot-dense')),
])

preprocess_pipeline = FeatureUnion(transformer_list=[
    ('numerical_pipeline', numerical_pipeline),
    ('categorical_pipeline', categorical_pipeline)
])

y = bank_rec['y']
X = bank_rec.drop(['y'], axis=1)
X = preprocess_pipeline.fit_transform(X)
print(pd.DataFrame(X).head())
