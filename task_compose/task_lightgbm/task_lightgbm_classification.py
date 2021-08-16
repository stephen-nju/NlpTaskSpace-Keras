# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:  18073701@suning.com
@software: PyCharm
@file: task_lightgbm_classification.py
@time: 2021/8/4 17:58
"""
# 使用lightgbm 对可替代品类进行过滤
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, KFold, GridSearchCV

import lightgbm as lgb
import numpy as np
import pandas as pd

# 加载训练集
df_train = pd.read_csv("")
feature_names = [c for c in df_train.columns if c not in ["category_one", "category_two", "label"]]
X_train = df_train[feature_names]
y_train = df_train["label"]
X_test = df_train[feature_names]
y_test = df_train["label"]

# 创建成lgb特征的数据集格式

lgb_train = lgb.Dataset(X_train, y_train, categorical_feature="auto")
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# # 将参数写成字典下形式
params = {
    'task': 'train',
    'boosting_type': 'gbdt',  # 设置提升类型
    'objective': 'binary',  # 目标函数
    'metric': {'l2', 'logloss'},  # 评估函数
    'num_leaves': 31,  # 叶子节点数
    'learning_rate': 0.05,  # 学习速率
    'feature_fraction': 0.9,  # 建树的特征选择比例
    'bagging_fraction': 0.8,  # 建树的样本采样比例
    'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
    'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
}

# # 训练 cv and train
gbm = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=lgb_eval, early_stopping_rounds=5)
#
# 保存模型到文件
# gbm.save_model('model.txt')
# # 预测数据集
y_pred_raw = gbm.predict(X_test)
y_pred = np.where(y_pred_raw < 0.5, 0, 1)
print(classification_report(y_test, y_pred))  # # 评估模型
# print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

# # feature importances
# print('Feature importances:', list(gbm.feature_importances_))


# # 网格搜索，参数优化
# estimator = lgb.LGBMClassifier(num_leaves=31)
# param_grid = {
#     'learning_rate': [0.01, 0.1, 1],
#     'n_estimators': [20, 40]
# }
# gbm = GridSearchCV(estimator, param_grid)
# gbm.fit(X_train, y_train)
# print('Best parameters found by grid search are:', gbm.best_params_)
