import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


dataset1 = pd.read_csv('test.csv')
dataset2 = pd.read_csv('sample_submission.csv')
dataset3 = pd.read_csv('train.csv')

X = dataset3.iloc[:, 1:-1].values
y = pd.DataFrame(dataset3, columns = ['SalePrice'])

X_test = pd.DataFrame(dataset1, columns=['LotArea', 'YrSold', 'YearBuilt']).values
y_test = pd.DataFrame(dataset2, columns = ['SalePrice'])
y_test = dataset2.iloc[:, -1].values

sns.heatmap(dataset3.corr(), xticklabels=1, yticklabels=1)
plt.show()

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,:])
X[:, :] = imputer.transform(X[:, :])


for i in range(0,79):
    if isinstance(X[i][0],str):
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        labelencoder_X = LabelEncoder()
        X[:, i] = labelencoder_X.fit_transform(X[:, i])
        onehotencoder = OneHotEncoder(categorical_features = [i])
        X = onehotencoder.fit_transform(X).toarray()



from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

y_pred = regressor.predict(X_test)

y_pred = sc_y.inverse_transform(y_pred)
