from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Generalizar -> Scaling(data_train,data_test,columns,metodo):

def Scaling_StandardScaler(data_train, data_test, columns):
    SC = StandardScaler()

    scaling_data = SC.fit_transform(data_train)
    scaling_data_train = pd.DataFrame(scaling_data, columns=columns)

    scaling_data_test = SC.transform(data_test)

    return scaling_data_train, scaling_data_test


def Scaling_MinMaxScaler(data_train, data_test, columns):
    MMS = MinMaxScaler()

    scaling_data = MMS.fit_transform(data_train)
    scaling_data_train = pd.DataFrame(scaling_data, columns=columns)

    scaling_data_test = MMS.transform(data_test)

    return scaling_data_train, scaling_data_test


def Scaling_RobustScaler(data_train, data_test, columns):
    Roboust = RobustScaler()

    scaling_data = Roboust.fit_transform(data_train)
    scaling_data_train = pd.DataFrame(scaling_data, columns=columns)

    scaling_data_test = Roboust.transform(data_test)
    return scaling_data_train, scaling_data_test



def ScalingComparationScaling(data, scaling_data):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))
    columns = list(data)

    ax1.set_title('Before Scaling')
    ax2.set_title('After Scaling')
    for name in columns:
        sns.kdeplot(data[name], ax=ax1)
        sns.kdeplot(scaling_data[name], ax=ax2)
    plt.show()
    pass
    
