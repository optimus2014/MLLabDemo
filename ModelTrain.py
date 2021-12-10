"""
需要实现的功能：
1.切分训练集、验证集、测试集（交叉验证）
2.特征工程，字段转化，连续型分箱等
    one-hot
    label encoding
    脏数据剔除，离群点等
3.解决数据不平衡问题，上取补充
4.评分体系：AUC，KS(待定)
5.网格搜索法训练模型
6.输出模型文件
7.测试预测函数
"""

import pandas as pd

from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit

# 导入 train_test_split
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance


class ModelTrain:
    """
    模型训练主类
    """
    # 待训练文件
    _data = None
    # 标签名称
    _target = None
    # 模型文件名
    _model_name = None

    # 模型
    _model = XGBClassifier(
            objective='binary:logistic',
            booster='gbtree',
            eval_metric='auc',  # auc
            use_label_encoder = False,
            nthread=-1,  # -1是使用全部线程,
            alpha=0.1,
            gamma=0.5,
            colsample_bytree=0.5,  # 经验值，通常是0.5
            scale_pos_weight=1,
            n_estimators=50,
            learning_rate=0.8,  # 经验值
        )
    # 缓存最佳模型
    _best_model = _model

    #
    _train_x = None
    _train_y = None

    _val_x = None
    _val_y = None

    def load_data(self,train_data : pd.DataFrame,target : str):
        """
        导入训练集文件
        :param train_data:训练时呼叫
        :param Target:标签项
        :return:
        """
        self._data = train_data
        self._target = target
        # 打印数据基础情况
        print("***************** 加载训练数据 *******************")
        print('训练集基础情况：', train_data.shape)
        # 正负样本分布情况
        print('正负样本分布情况：')
        label_dist = self._data[target].groupby(self._data[target]).count().sort_values(ascending=False)
        print(label_dist)

        # 解决数据分布不平衡问题，当正负样本差小于25%，触发balance操作
        # TODO：逻辑待定

        # 切分数据
        train_x, val_x, train_y, val_y = self.split_data(self._data)
        # 训练集
        self._train_x = train_x
        self._train_y = train_y
        # 校验集
        self._val_x = val_x
        self._val_y = val_y

    def train(self,model_name = "default"):
        """
        训练模型
        保存模型文件
        :return:
        """
        self._model_name = model_name
        param = self._gridSearch()
        print("待补充模型参数：" + str(param))
        # 配置最佳参数，训练最终模型
        self._best_model_train(param)
        self._save_model(model_name)
        # self.show_model()
        self.predict(None)


    def show_model(self):
        """
        终端输出模型参数
        打印模型信息log
        :return:
        """
        pass


    def predict(self,test_data) -> float:
        """
        预测模型
        打印模型评分情况
        :param test_data:测试集
        :return: 评分
        """

        local_model = xgb.Booster({'nthread': 4})  # init model
        local_model.load_model(self._model_name)
        test_pred = local_model.predict(xgb.DMatrix(self._val_x))  #, iteration_range=self._best_model.best_iteration

        result = pd.DataFrame(data=test_pred,
                              columns=['target'])  # .applymap(lambda i: 1 if i >= 0.5 else 0) # 对概率做二分类转化
        fpr, tpr, thresholds = roc_curve(self._val_y, result, pos_label=1)
        print("阈值：{value}，AUC:{auc}".format(value=thresholds, auc=auc(fpr, tpr)))



    def split_data(self,data : pd.DataFrame,target:str = None):
        """
        切分数据集,默认训练集：验证集为8：2
        注：初始将原始数据切分为训练集和最终的验证集
        :param data:原始完整的数据集
        :return:返回训练集和测试集，其中测试集不参与训练
        """
        if target is None:
            target = self._target
        train_x, val_x, train_y, val_y = train_test_split(data.drop([target], axis=1).reset_index(drop=True),
                                                          data[target], test_size=0.2, random_state=100)

        train = pd.concat([train_x, train_y], axis=1)
        train.reset_index(drop=True, inplace=True)
        val = pd.concat([val_x, val_y], axis=1)
        val.reset_index(drop=True, inplace=True)

        return (train.drop([target], axis=1), val.drop([target], axis=1), train[target], val[target])

    def _gridSearch(self)-> dict:
        """
        网格搜索法
        :return:
        """
        train_x = self._train_x
        train_y = self._train_y

        # 定义交叉验证
        kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=8707111)
        # 网格搜索法，计算最佳参数集合
        param_grid = dict(
            min_child_weight=list(range(20)),
            max_depth=list(range(2, 10)),
        )

        grid_search = GridSearchCV(self._model, param_grid, scoring="roc_auc", n_jobs= -1,   #-1,
                                   cv=kfold,  verbose=1)
        print("************* 开始进行网格搜索 ************")
        grid_result = grid_search.fit(train_x, train_y)
        print("Best:%f using %s" % (grid_result.best_score_, grid_result.best_params_))

        # print(self._model.best_score)
        # print(self._model.max_depth)

        # means = grid_result.cv_results_['mean_test_score']
        # stds = grid_result.cv_results_['std_test_score']
        # params = grid_result.cv_results_['params']
        # for mean, stdev, param in zip(means, stds, params):
        #     print("%f (%f) with: %r" % (mean, stdev, param))
        #
        return grid_result.best_params_

    def _best_model_train(self,param : dict):
        self._best_model.max_depth = param["max_depth"]
        self._best_model.min_child_weight = param["min_child_weight"]

        # 拆分训练集和测试集

        sub_train_x, test_x, sub_train_y, test_y = self.split_data(pd.concat([self._train_x, self._train_y], axis=1))

        # 训练最佳参数
        self._best_model.fit(sub_train_x, sub_train_y,
                             early_stopping_rounds=66,
                             eval_set=[(test_x, test_y)])



    def _save_model(self,model_name = None):
        if model_name is None:
            model_name = self._model_name

        self._model_name = './' + model_name + '.model'
        self._best_model.save_model(self._model_name)

        # 打印特征重要性
        print("特征重要性：" + str(plot_importance(self._best_model)))
        # pyplot.show()


