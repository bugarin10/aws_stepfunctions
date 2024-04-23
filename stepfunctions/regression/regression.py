import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def lambda_handler(event, context):
    x_train = event["x_train"]
    x_test = event["x_test"]
    y_train = event["y_train"]
    y_test = event["y_test"]

    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(x_train, y_train)

    # Predict on the test data
    y_pred = model.predict(x_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Return the accuracy as the response
    return {
        'statusCode': 200,
        'body': json.dumps({'accuracy': accuracy})
    }