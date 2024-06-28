from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open('Machine_failure.pkl', 'rb') as file:
    loaded_object = pickle.load(file)

# If the loaded object is a tuple, extract the model
if isinstance(loaded_object, tuple):
    model = loaded_object[0]
else:
    model = loaded_object

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    type_ = data['type']
    air_temperature = data['air_temperature']
    process_temperature = data['process_temperature']
    rotational_speed = data['rotational_speed']
    torque = data['torque']
    tool_wear = data['tool_wear']

    # Prepare the feature array for prediction
    features = np.array([[type_, air_temperature, process_temperature, rotational_speed, torque, tool_wear]])
    
    # Make prediction
    prediction = model.predict(features)
    
    result = bool(prediction[0])

    return jsonify({'machine_failure': result})

if __name__ == '__main__':
    app.run(debug=True)

