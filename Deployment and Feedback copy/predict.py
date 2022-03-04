import pandas as pd 
import numpy as np 
import pickle

from sklearn.metrics import mean_squared_error, r2_score


# load the dataset
df_cars_test = pd.read_csv('bmw_test.csv')

# load the model
model_filename = 'final_model.sav'
model = pickle.load(open(model_filename, 'rb'))

#predict
X_test = df_cars_test.drop('price', axis=1)
y_test = df_cars_test['price']

y_pred = model.predict(X_test)

df_cars_test['price_pred'] = y_pred
df_cars_test.to_csv('pred_result.csv', index=False)

print('RMSE :', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R-squared :', r2_score(y_test, y_pred))