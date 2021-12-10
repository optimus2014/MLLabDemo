"""
类说明：
计算各个特征的得分
分别有
woe
iv

辅助功能有：
1.检查对象是否符合规范，check()
2.记录最终得分情况
3.连续型特征，进行分箱操作
4.异常值处理，缺失值、明显偏离值等。
5.哑变量处理


分箱方法：
1.one-hot：离散型变量
2.有监督算法，卡方分箱

"""

import numpy as np
import pandas as pd
import math

from sklearn.tree import DecisionTreeClassifier


class FeatureScore:
    # 载入数据
    _df = None
    # 标签名称，目前只适配二分类，标签只能是0,1
    _target = None
    # 需要计算的数据列
    _cols = None
    # 最终的特征得分
    _score = dict()

    def __init__(self,data:pd.DataFrame ,target:str):
        self._df = data
        self._cols = data.columns
        self._target = target
        print(data.shape)
        data.head()

    def cal(self,col: str, is_box = False ,box_points = None) -> pd.DataFrame:
        '''
        计算变量各个分箱的WOE、IV值，返回一个DataFrame
        :return:
        '''
        print(f" ******************** 特征：{col} 计算开始 ********************** ")

        if col not in self._cols:
            print(f"ERROR:特征{col}不存在.")
            print(f" ******************** 特征：{col} 计算结束 ********************** \n\n")
            return


        # 如果需要分箱，获取分箱之后的结果，
        if is_box:
            df = self.trans_box(col,boundary = box_points)
        else:
            x = self._df[col]
            x = x.fillna("NULL")  # 填充缺失值
            # 对应标签结果
            y = self._df[self._target]
            df = pd.concat([x, y], axis=1)  # 合并x、y为一个DataFrame，方便后续计算
            df.columns = ['x', 'y']  # 特征变量、目标变量字段的重命名

        grouped = df.groupby('x')['y']  # 统计各分箱区间的好、坏、总客户数量
        result_df = grouped.agg([('good', lambda y: (y == 0).sum()),
                                 ('bad', lambda y: (y == 1).sum()),
                                 ('total', 'count')])
        result_df['good_pct'] = result_df['good'] / result_df['good'].sum()  # 好客户占比
        result_df['bad_pct'] = result_df['bad'] / result_df['bad'].sum()  # 坏客户占比
        result_df['total_pct'] = result_df['total'] / result_df['total'].sum()  # 总客户占比

        result_df['bad_rate'] = result_df['bad'] / result_df['total']  # 坏比率

        result_df['woe'] = np.log(result_df['good_pct'] / result_df['bad_pct'])  # WOE
        result_df['iv'] = (result_df['good_pct'] - result_df['bad_pct']) * result_df['woe']  # IV
        result_df.replace([np.inf, -np.inf], 0, inplace=True)
        result_df.replace(np.nan, 0, inplace=True)
        self._score[col] = result_df
        self._show(result_df)
        print(f" ******************** 特征：{col} 计算结束 ********************** \n\n")
        return result_df

    def trans_box(self,col: str, nan: float = -999., boundary = None) -> pd.DataFrame:
        x = self._df[col]
        x = x.fillna(nan)  # 填充缺失值
        # 对应标签结果
        y = self._df[self._target]

        # 有分箱的数据分割点，即使用，没有的话使用默认的决策树方法进行分箱
        if boundary is None:
            boundary = self._box(col, y)  # 获得最优分箱边界值列表
        # else:
        #     boundary = boundary

        print("分箱结果为：" + str(boundary))

        df = pd.concat([x, y], axis=1)  # 合并x、y为一个DataFrame，方便后续计算
        df.columns = ['x', 'y']  # 特征变量、目标变量字段的重命名
        df['bins'] = pd.cut(x=x, bins=boundary, right=False)  # 获得每个x值所在的分箱区间
        df = df.drop(labels='x', axis=1)
        df.rename(columns={'bins': 'x'}, inplace=True)
        return df

    def save(self,file_name = "result.xlsx"):
        '''
        特征得分的结果文件，保存到指定文件中
        :return:
        '''
        # print(self._score)
        # self._score["Age"].to_csv(file_name)
        writer = pd.ExcelWriter(file_name)
        for key in self._score.keys():
            self._score[key].to_excel(writer, sheet_name=key)
        writer.save()
        writer.close()


    def _box(self,col: str, nan: float = -999.) -> list:
        '''
        利用决策树获得最优分箱的边界值列表，
        :param col:列名称
        :return:返回分箱点的数组
        '''
        boundary = []  # 待return的分箱边界值列表
        # 待分箱列
        x = self._df[col]
        x = x.fillna(nan).values  # 填充缺失值
        # 对应标签结果
        y = self._df[self._target].values
        clf = DecisionTreeClassifier(criterion='entropy',  # “信息熵”最小化准则划分
                                     max_leaf_nodes=6,  # 最大叶子节点数
                                     min_samples_leaf=0.05)  # 叶子节点样本数量最小占比

        clf.fit(x.reshape(-1, 1), y)  # 训练决策树

        n_nodes = clf.tree_.node_count
        children_left = clf.tree_.children_left
        children_right = clf.tree_.children_right
        threshold = clf.tree_.threshold

        for i in range(n_nodes):
            if children_left[i] != children_right[i]:  # 获得决策树节点上的划分边界值
                boundary.append(threshold[i])

        boundary.sort()

        min_x = x.min()
        max_x = x.max() + 0.1  # +0.1是为了考虑后续groupby操作时，能包含特征最大值的样本
        boundary = [min_x] + boundary + [max_x]
        return boundary


    def _show(self,result_df):
        '''
        打印特征得分结果
        :return:
        '''
        print(result_df)
        print(f"该变量IV = {result_df['iv'].sum()}")

    def woe(self):
        '''
        计算特征woe值
        :return:
        '''
        pass

    def iv(self):
        '''
        计算特征iv值
        :return:
        '''
        pass
