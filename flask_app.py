from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load your models and preprocessor
decision_tree = pickle.load(open('models/decisiontreeregressor.pkl', 'rb'))
linear_regression = pickle.load(open('models/regression_model (6).pkl', 'rb'))
lasso = pickle.load(open('models/final_lasso (5).pkl', 'rb'))
ridge = pickle.load(open('models/final_ridge (4).pkl', 'rb'))
preprocessor = pickle.load(open('models/preprocessor.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(request.form.get(feature)) for feature in ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']]
    city = request.form.get('City')  # Assuming 'City' is a category that you may handle separately
    features = np.array(features).reshape(1, -1)
    scaled_features = preprocessor.transform(features)

    prediction1 = decision_tree.predict(scaled_features)[0]
    prediction3 = linear_regression.predict(scaled_features)[0]
    prediction4 = lasso.predict(scaled_features)[0]
    prediction5 = ridge.predict(scaled_features)[0]

    return render_template('results.html',
                           prediction1=prediction1,
                           prediction3=prediction3,
                           prediction4=prediction4,
                           prediction5=prediction5)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
