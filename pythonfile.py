import json
import joblib
import numpy as np

# Load the model
model = joblib.load("decision_tree_model.joblib")

def lambda_handler(event, context):
    # Parse the input from the event
    body = json.loads(event['body'])
    feature1 = body['feature1']
    feature2 = body['feature2']
    
    # Create the input data for the model
    input_data = np.array([[feature1, feature2]])
    
    # Make the prediction
    prediction = model.predict(input_data)[0]
    
    # Create the response
    response = {
        'statusCode': 200,
        'body': json.dumps({'prediction': int(prediction)})
    }
    
    return response
