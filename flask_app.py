from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the models
try:
    with open('models/decisiontreeregressor.pkl', 'rb') as f:
        decision_tree_regressor = pickle.load(f)
    with open('models/linearregression.pkl', 'rb') as f:
        linear_regression = pickle.load(f)
    with open('models/lasso.pkl', 'rb') as f:
        lasso = pickle.load(f)
    with open('models/ridge.pkl', 'rb') as f:
        ridge = pickle.load(f)
    with open('models/preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        df = pd.DataFrame([data])

        df_processed = preprocessor.transform(df)

        prediction_tree = decision_tree_regressor.predict(df_processed)
        prediction_linear = linear_regression.predict(df_processed)
        prediction_lasso = lasso.predict(df_processed)
        prediction_ridge = ridge.predict(df_processed)

        predictions = {
            'Decision Tree': prediction_tree.tolist(),
            'Linear Regression': prediction_linear.tolist(),
            'Lasso': prediction_lasso.tolist(),
            'Ridge': prediction_ridge.tolist()
        }

        return jsonify(predictions)
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({"error": "An error occurred during prediction."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
