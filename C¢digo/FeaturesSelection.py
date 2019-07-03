from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
import matplotlib.pyplot as plt




def FeaturesSelection_US(data, target, k):
    # apply SelectKBest class to extract top k best features
    bestfeatures = SelectKBest(score_func=chi2, k=k)
    fit = bestfeatures.fit(data, target)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(data.columns)
    # concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    # print(featureScores.nlargest(16, 'Score'))

    return featureScores.nlargest(k, 'Score')['Specs']


def Univariate_Selection(dataset, target, n_features):
    best_features = FeaturesSelection_US(dataset, target, n_features)
    dataset_selected = dataset[best_features.values]
    return dataset_selected


def FeaturesSelection_Extra_Trees_Classifier(data, target, k):
    model = ExtraTreesClassifier()
    model.fit(data, target)
    # print(model.feature_importances_)  # use inbuilt class feature_importances of tree based classifiers
    # plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=data.columns)
    feat_importances.nlargest(k).plot(kind='barh')
    plt.show()
    return feat_importances.nlargest(k)


def Feature_Importance(dataset, target, n_features):
    # Feature Importance with Extra Trees Classifier)-(mais alto melhor relação)
    best_features = FeaturesSelection_Extra_Trees_Classifier(dataset, target, n_features)
    dataset_selected = dataset[best_features.index]
    return dataset_selected
