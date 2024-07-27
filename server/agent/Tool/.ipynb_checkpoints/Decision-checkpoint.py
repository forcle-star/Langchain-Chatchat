# 预处理输入的数据，然后加载模型，并执行即可
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.stats import  skew
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)  # 把所有基准模型的预测结果作为（元特征）用于训练元模型
        return self

    # Do the predictions of all base models on the test data and use the averaged predictions as
    # meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)

def data_process(input_data):
    # 读用于辅助预处理的数据
    #train = pd.read_excel('/home/lee/lhd/lhd/Langchain-Chat2.0/server/agent/Tool/simulated_datasets.xlsx')
    train = pd.read_excel('/root/Langchain-Chatchat/server/agent/Tool/simulated_datasets.xlsx')
    keys = list(input_data.keys())
    values = list(input_data.values())
    test = pd.DataFrame(columns=keys)
    test.loc[0] = values
    # 删除离谱的数据
    train = train.drop(train[(train['TopDisplacement']>1000) & (train['WaistDisplacement']>600) & (train['BottomDisplacement']>2000)].index)

    # 获取现在train和test的数据量:
    ntrain = train.shape[0]
    ntest = test.shape[0]

    # 将训练集和测试集合并在一起进行预处理处理，后续根据ntrain可以将两者完美分开
    all_data = pd.concat((train, test)).reset_index(drop=True)
    all_data.drop(['TopDisplacement'], axis=1, inplace=True)
    all_data.drop(['WaistDisplacement'], axis=1, inplace=True)
    all_data.drop(['BottomDisplacement'], axis=1, inplace=True)

    # 处理缺失值问题
    columns_to_fill = ['StepHeight', 'StepLength', 'SmallPipeAngle', 'SmallPipeSpacing','SmallPipeDiameter',
                       'AnchorRodAngle', 'AnchorRodSpacing', 'AnchorRodDiameter','AnchorRodVerticalSpacing', 'AnchorRodLength']
    all_data[columns_to_fill] = all_data[columns_to_fill].fillna(0)  # 有的数据不存在这些列是合理的

    # 对类别类型的数据进行独热编码
    cols = ('SectionType', 'SecondarySupportStrengthGrade', 'InitialSupportGrade', 'ArchType', 'ExcavationMode')
    for c in cols:
        # print("处理前的数据%s：".format(c), all_data[c])
        lbl = LabelEncoder()
        lbl.fit(list(all_data[c].values))
        all_data[c] = lbl.transform(list(all_data[c].values)) + 1


    # 特征feature具有偏斜的程度 检查数值特征的偏斜程度(后续考虑是否需要对偏斜程度大于0.7做一下处理？)
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

    # Check the skew of all numerical features
    skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness = pd.DataFrame({'偏斜程度Skew' :skewed_feats})

    skewness = skewness[abs(skewness['偏斜程度Skew']) > 0.75]

    from scipy.special import boxcox1p
    skewed_features = skewness.index
    lam = 0.15
    for feat in skewed_features:
        all_data[feat] = boxcox1p(all_data[feat], lam)

    # 对数据进行归一化
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler() # 创建 MinMaxScaler 实例
    all_data_scaled = all_data.copy()  # 创建一个副本用于存储归一化后的数据
    for column in all_data.columns:  # 对数据集的每个列进行归一化
        all_data_scaled[column] = scaler.fit_transform(all_data[[column]])
    all_data = all_data_scaled

    # 刻舟求剑，将处理好的数据划分为训练集和测试集
    # train = all_data[:ntrain]
    test = all_data[ntrain:]
    return test

import pickle
import os
import numpy as np
from typing import Type
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

class DecisionInput(BaseModel):
    X_test: str = Field(description="The input data to Predict")

class Decision(BaseTool):
    name = "Decision"
    description = "Predict the corresponding Bottom Displacement based on the given parameter list"
    args_schema: Type[BaseModel] = DecisionInput

    def __init__(self):
        super().__init__()

    def _run(self, X_test: str) -> str:
        # Ensure the DecisionTree.pickle file exists
        #if not os.path.exists('/home/lee/lhd/lhd/Langchain-Chat2.0/server/agent/Tool/stacked_averaged_models.pkl'):
        if not os.path.exists('/root/Langchain-Chatchat/server/agent/Tool/stacked_averaged_models.pkl'):
            raise Exception("stacked_averaged_models.pkl file is missing")

        # Load the trained classifier
        #with open('/home/lee/lhd/lhd/Langchain-Chat2.0/server/agent/Tool/stacked_averaged_models.pkl', 'rb') as f:
        with open('/root/Langchain-Chatchat/server/agent/Tool/stacked_averaged_models.pkl', 'rb') as f:
            stacked_averaged_models = pickle.load(f)

        must_keys_list = ['SecondarySupportRatio', 'InternalFrictionAngle', 'InitialSupportRatio',
                          'SectionType', 'ElasticModulus', 'TensileStrength', 'PoissonsRatio',
                          'Cohesion', 'HorizontalStress', 'LongitudinalStress', 'VerticalStress',
                          'SecondarySupportThickness', 'SecondarySupportStrengthGrade', 'InitialSupportThickness',
                          'InitialSupportGrade', 'SmallPipeAngle', 'SmallPipeSpacing', 'SmallPipeDiameter',
                          'ArchType', 'ArchSpacing', 'ExcavationMode', 'ExcavationLength', 'StepHeight',
                          'StepLength', 'TopDisplacement', 'WaistDisplacement']
        all_keys_list = ['SecondarySupportRatio', 'InternalFrictionAngle',
                         'InitialSupportRatio', 'SectionType', 'ElasticModulus',
                         'TensileStrength', 'PoissonsRatio', 'Cohesion', 'HorizontalStress',
                         'LongitudinalStress', 'VerticalStress', 'SecondarySupportThickness',
                         'SecondarySupportStrengthGrade', 'InitialSupportThickness', 'InitialSupportGrade',
                         'SmallPipeAngle', 'SmallPipeSpacing', 'SmallPipeDiameter', 'ArchType', 'ArchSpacing',
                         'AnchorRodAngle', 'AnchorRodSpacing', 'AnchorRodDiameter', 'AnchorRodVerticalSpacing',
                         'AnchorRodLength', 'ExcavationMode', 'ExcavationLength', 'StepHeight',
                         'StepLength', 'BottomDisplacement', 'TopDisplacement', 'WaistDisplacement']
        out_data = {}
        f_data = {}

        for k_v in X_test.split(', '):
            key_value = k_v.split(': ')
            key = key_value[0]
            value = float(key_value[1])
            out_data[key] = value

        for key_k in all_keys_list:
            if key_k in out_data.keys():
                must_keys_list.append(key_k)
                f_data[key_k] = out_data[key_k]
            elif key_k in must_keys_list:
                f_data[key_k] = 0
            else:
                f_data[key_k] = None
        # Predict the class of the input data
        test = data_process(f_data)
        y_pred = np.expm1(stacked_averaged_models.predict(test.values))
        return str(y_pred[0])

# if __name__ == "__main__":
#     decision = Decision()
#     input_data =("SecondarySupportRatio: 0.233732, InternalFrictionAngle: 30.53334,"
#                  " InitialSupportRatio: 0.443486, SectionType: 1, ElasticModulus: 2.01790069026405, "
#                  "TensileStrength: 2.925326178086, PoissonsRatio: 0.291758, Cohesion: 0.748672297836, "
#                  "HorizontalStress: 3.0662178087, "
#                  "LongitudinalStress: 5.1823492669, VerticalStress: 7.443241505, "
#                  "SecondarySupportThickness: 45, SecondarySupportStrengthGrade: 2, "
#                  "InitialSupportThickness: 3, InitialSupportGrade: 2, SmallPipeAngle: 30, "
#                  "SmallPipeSpacing: 0.5, SmallPipeDiameter: 42, ArchType: 15, "
#                  "ArchSpacing: 0.8, ExcavationMode: 2, ExcavationLength: 4, StepHeight: 4.046")
#
#     result = decision.run(input_data)
#     print(result)