import pandas as pd

def tree():
    # 导入数据集
    data = "/home/xly/data/znjz/classify_pro.xlsx"

    iris_local = pd.read_excel(data)
    iris_local = iris_local.dropna()  # 丢弃含空值的行、列
    # 载入特征和标签集
    X = iris_local[['坚硬程度', '完整程度', '岩层厚度', '地下水量', '埋深水平', '地质构造', '施工工法',
                    '内轮廓形式']]  # 等价于iris_dataset.data
    y = iris_local['支护方案']  # 等价于iris_dataset.target
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import metrics

    model = DecisionTreeClassifier(max_depth=12, min_samples_leaf=2, criterion='gini')
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    print('The accuracy of the Decision Tree is: {:.3f}'.format(metrics.accuracy_score(prediction, y_test)))
    import pickle
    import numpy as np

    with open('DecisionTree.pickle', 'wb') as f:
        pickle.dump(model, f)  # 将训练好的模型clf存储在变量f中，且保存到本地
    with open('DecisionTree.pickle', 'rb') as f:
        clf_load = pickle.load(f)  # 将模型存储在变量clf_load中
        nd = np.array([[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]])
        prediction = clf_load.predict(nd)
        print(prediction)

from typing import Type
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import pickle
import os
import numpy as np


class ClassificationInput(BaseModel):
    X_test: str = Field(description="The input data to classify")


class Classifier(BaseTool):
    name = "Classifier"
    description = "Use a trained Decision Tree classifier to classify input data"
    args_schema: Type[BaseModel] = ClassificationInput

    def __init__(self):
        super().__init__()

    def _run(self, X_test: str) -> str:
        # Ensure the DecisionTree.pickle file exists
        if not os.path.exists('./Tool/DecisionTree.pickle'):
            raise Exception("./Tool/DecisionTree.pickle file is missing")

        # Load the trained classifier
        with open('./Tool/DecisionTree.pickle', 'rb') as f:
            clf_load = pickle.load(f)

        import re
        numbers = re.findall(r'\d+', X_test)
        # 将找到的数字字符串转换为整数
        X = [int(num) for num in numbers]
        print(X)
        # Predict the class of the input data
        import numpy as np
        nd = np.array(X)
        nd = nd.reshape((1, -1))
        prediction = clf_load.predict(nd)

        return str(prediction[0])

if __name__ == "__main__":
    tree()
    classfier = Classifier()
    result = classfier.run("[1,2,3,4,5,6,7,8]")
    print(result)