# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
from FeatureScore import FeatureScore

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = pd.read_csv('titanic_data.csv')
    print("******************* START *****************")
    fscore = FeatureScore(data,"Survived")
    fscore.cal("SibSp", is_box=False)
    fscore.cal("Age",is_box = True)
    fscore.cal("Age", is_box=True,box_points=[0,10,50,70,150])
    fscore.save()
