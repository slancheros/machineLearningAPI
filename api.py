# Dependencies
from flask import Flask, request, jsonify
from sklearn.externals import joblib
import sys
import traceback
import pandas as pd


# Your API definition
app = Flask(__name__)

@app.route('/')
def hello():
    return "Hola, bienvenidos al API de ML!"

@app.route('/predict', methods=['POST'])
def predict():
    if lr:
        try:
            json_ = request.json
            query = pd.get_dummies(pd.DataFrame(json_))
            print(pd.DataFrame(json_))
            print(query)
            query = query.reindex(columns=model_columns, fill_value=0)
            print(query)
            prediction = list(lr.predict(query))
            return str(prediction)
            #return jsonify({'prediction': str(prediction)})

        except BaseException:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print('Train the model first')
        return 'No model here to use'


if __name__ == '__main__':
    try:
        port = int(sys.argv[1])  # This is for a command-line input
    except BaseException:
        port = 12345  # If you don't provide any port the port will be set to 12345

    lr = joblib.load('model.pkl')
    print('Model loaded')

    model_columns = joblib.load('model_columns.pkl')
    print('Model columns loaded')
    app.run(port=port, debug=True)
