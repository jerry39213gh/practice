from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
import numpy as np

boston = load_boston()
# House Prices
y = boston.target
# The other 13 features
x = boston.data

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

rf = RandomForestRegressor(n_estimators=100,
                            n_jobs=-1,
                            random_state=1)

gdbr = GradientBoostingRegressor(learning_rate=0.1,
                                  loss='ls',
                                  n_estimators=100,
                                  random_state=1)

abr = AdaBoostRegressor(DecisionTreeRegressor(),
                         learning_rate=0.1,
                         loss='linear',
                         n_estimators=100,
                         random_state=1)

for i in [rf, gdbr, abr]:
    cross_val_score(i, x, y, cv=5, scoring='r2')
for i in [rf, gdbr, abr]:
    cross_val_score(i, x, y, cv=5, scoring='mean_squared_error')

gdbr1 = GradientBoostingRegressor(learning_rate=1,
                                  loss='ls',
                                  n_estimators=100,
                                  random_state=1)


def stage_score_plot(estimator, x_train, y_train, x_test, y_test):
    '''
    Parameters: estimator: GradientBoostingRegressor or AdaBoostRegressor
                X_train: 2d numpy array
                y_train: 1d numpy array
                X_test: 2d numpy array
                y_test: 1d numpy array

    Returns: A plot of the number of iterations vs the MSE for the model for
    both the training set and test set.
    '''
    estimator.fit(x_train, y_train)
    train_mse = []
    test_mse = []
    for train_pred, test_pred in zip(estimator.staged_predict(x_train), estimator.staged_predict(x_test)):
        train_mse.append(mse(train_pred, y_train))
        test_mse.append(mse(test_pred, y_test))
    plt.plot(train_mse, label='training mse')
    plt.plot(test_mse, label='test mse')
    plt.legend()


### grid search
random_forest_grid = {'max_depth': [3, None],
                      'max_features': ['sqrt', 'log2', None],
                      'min_samples_split': [2, 4],
                      'min_samples_leaf': [1, 2, 4],
                      'bootstrap': [True, False],
                      'n_estimators': [10, 20, 40, 80],
                      'random_state': [1]}

rf_gridsearch = GridSearchCV(RandomForestRegressor(),
                             random_forest_grid,
                             n_jobs=-1,
                             verbose=True,
                             scoring='mean_squared_error')
rf_gridsearch.fit(X_train, y_train)

print("best parameters:", rf_gridsearch.best_params_)

best_rf_model = rf_gridsearch.best_estimator_
