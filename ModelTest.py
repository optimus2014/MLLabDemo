import pandas as pd
from ModelTrain import ModelTrain

if __name__ == '__main__':
    print("模型测试主函数")
    # ********加载数据*******
    titan = pd.read_csv('titanic_data.csv')

    # *********** 特征工程 *************
    # label econding特征处理
    code = pd.factorize(titan['Sex'])
    titan['Sex'] = pd.Series(code[0])

    code = pd.factorize(titan['Embarked'])
    titan['Embarked'] = pd.Series(code[0])

    # 剔除无效特征
    titan = titan.drop(["Name"], axis=1).drop(["Ticket"], axis=1).drop(["Cabin"], axis=1)

    # 填充空缺值
    titan.fillna(0)

    # ********** 模型训练 **********
    model = ModelTrain()
    model.load_data(titan, "Survived")
    model.train("titanic模型")

