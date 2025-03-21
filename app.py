from flask import Flask, render_template, request
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model
model_filename = 'finalized_model.sav'
model = pickle.load(open(model_filename, 'rb'))

# Initialize Flask app
app = Flask(__name__)

# Load the scaler
scaler = StandardScaler()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input features from form
        input_data = [float(x) for x in request.form.values()]

        # Convert input into a NumPy array and reshape it
        input_array = np.array(input_data).reshape(1, -1)

        # Standardize input data
        scaled_input = scaler.fit_transform(input_array)

        # Predict using the model
        prediction = model.predict(scaled_input)

        # Interpret result
        result = "Mine" if prediction[0] == 1 else "Rock"

        return render_template('index.html', prediction_text=f'Result: {result}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')


if __name__ == '__main__':
    app.run(debug=True)
