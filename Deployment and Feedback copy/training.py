import numpy as np
import pandas as pd
import pickle
import warnings

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings('ignore')

# load the training dataset
df_cars = pd.read_csv('bmw_train.csv')

# data preprocessing
X = df_cars.drop('price', axis=1)
y = df_cars['price']

scaled_cols = ['mileage', 'tax', 'mpg']
onehot_encoded_cols = ['model', 'transmission', 'fuelType']
ordinal_encoded_cols = ['year', 'engineSize']

scaler = MinMaxScaler()
onehot_encoder = OneHotEncoder(handle_unknown='ignore')
ordinal_encoder = OrdinalEncoder(handle_unknown='ignore')

# model pipeline
col_transformer = ColumnTransformer([
    ('scaler', scaler, scaled_cols),
    ('onehot', onehot_encoder, onehot_encoded_cols),
    ('ordinal', ordinal_encoder, ordinal_encoded_cols)
])

model_pipeline = Pipeline([
    ('col_transformer', col_transformer),
    ('poly', PolynomialFeatures(degree=2)),
    ('regression', LinearRegression())
])

# fit and tune hyperparameter
param_grid = {
    'regression__fit_intercept' : [False, True],
    #'poly__degree' : [1,2]
}

grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1
)

grid_search.fit(X, y)

# retrain the optimal model
final_model = grid_search.best_estimator_.fit(X, y)

# pickle the model
pickle_filename = 'final_model.sav'
pickle.dump(final_model, open(pickle_filename, 'wb'))