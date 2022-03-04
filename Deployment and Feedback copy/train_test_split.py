import pandas as pd 
from sklearn.model_selection import train_test_split


# load the dataset
df_cars = pd.read_csv('bmw.csv')
X = df_cars.drop('price', axis=1)
y = df_cars['price']

X['model'] = X['model'].apply(lambda x: x.strip())

# split the data train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)



# save train and test into csv
df_cars_train = pd.concat([X_train, y_train], axis=1)
df_cars_test = pd.concat([X_test, y_test], axis=1)

df_cars_train.to_csv('bmw_train.csv', index=False)
df_cars_test.to_csv('bmw_test.csv', index=False)