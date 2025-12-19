from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    area = int(request.form['area'])
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])
    neighborhood = request.form['neighborhood']

    input_data = pd.DataFrame({
        'GrLivArea': [area],
        'BedroomAbvGr': [bedrooms],
        'FullBath': [bathrooms],
        'Neighborhood': [neighborhood]
    })

    prediction = model.predict(input_data)
    price = int(prediction[0])

    return render_template('index.html', prediction_text=f"Predicted House Price: ₹ {price}")

if __name__ == "__main__":
    app.run(debug=True)
