# 1. prepare the training set and test set
# 2. prepare training script (pickle)
# 3. prepare test scirpt
# 4. crate the flask app : page routing, actual prediction
# 5. UI



from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle 

app = Flask(__name__)

# home page
@app.route('/')
def home():
    return render_template('index.html')

# prediction page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    df_cars = pd.read_csv('bmw_train.csv')
    car_model = df_cars['model'].unique().tolist()
    car_year = sorted(df_cars['year'].unique().tolist())
    car_transmission = df_cars['transmission'].unique().tolist()
    car_fuel_type = df_cars['fuelType'].unique().tolist()
    car_engine_size = sorted(df_cars['engineSize'].unique().tolist())

    car_spec = {
        'model': car_model,
        'year': car_year,
        'transmission': car_transmission,
        'fuel_type': car_fuel_type,
        'engine_size': car_engine_size
    }

    return render_template('predict.html', spec=car_spec)

# result page
@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        input = request.form

        df_predict  = pd.DataFrame({
            'model': [input['bmw_model']],
            'year': [int(input['year'])],
            'transmission': [input['transmission']],
            'mileage': [int(input['mileage'])],
            'fuelType': [input['fuel_type']],
            'tax': [int(input['tax'])],
            'mpg': [float(input['mpg'])],
            'engineSize': [float(input['engine_size'])],
        })

        prediction = int(model.predict(df_predict)[0])

        return render_template('result.html', spec=input, pred_result=prediction)


if __name__ == '__main__':
    model_filename = 'final_model.sav'
    model = pickle.load(open(model_filename, 'rb'))

    app.run(debug=True)