from flask import Flask, jsonify, request
from flask_cors import CORS
from evalane_sim_pipeline import EvalaneSimPipeline
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Load the pipeline once when the server starts
pipeline = EvalaneSimPipeline.load("evalane_sim_pipeline.pkl")
print("Pipeline loaded successfully!")


@app.route('/')
def home():
    return jsonify({'message': 'Evalane API is running'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Build the observation from the request
        observation = {
            "Hour": data.get("Hour", datetime.now().hour),
           "Peak_Hour": 1 if datetime.now().hour in {7,8,9,17,18,19} else 0,
            "Lane_N_Count":       data.get("Lane_N_Count", 0),
            "Lane_S_Count":       data.get("Lane_S_Count", 0),
            "Lane_E_Count":       data.get("Lane_E_Count", 0),
            "Lane_W_Count":       data.get("Lane_W_Count", 0),
            "Total_Vehicles":     data.get("Total_Vehicles", 0),
            "Lane_Imbalance":     data.get("Lane_Imbalance", 0),
            "Dominant_Lane":      data.get("Dominant_Lane", "N"),
            
           "Avg_Speed": data.get("Avg_Speed", 
    10 if data.get("Event_Type") == "Breakdown" else 
    15 if data.get("Total_Vehicles", 0) > 20 else 20),
            "Dedicated_Lane":     data.get("Dedicated_Lane", 0),
            "Bypass_Active":      data.get("Bypass_Active", 0),
            "Clear_Zone_Active":  data.get("Clear_Zone_Active", 0),
            "Obstacle_Present":   data.get("Obstacle_Present", 0),
            "Event_Type":         data.get("Event_Type", "Normal"),
            "Event_Lane":         data.get("Event_Lane", None),
        }

        result = pipeline.predict(observation)
        return jsonify(result)

    except Exception as e:
        print("ERROR:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)