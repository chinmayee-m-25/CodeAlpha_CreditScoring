from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('best_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data     = request.get_json()
        input_df = pd.DataFrame([data])

        input_df['BALANCE_TO_INVEST'] = input_df['ACCCURRENTBALANCE'] / (input_df['INVESTMENT_TOTAL'] + 1)
        input_df['PAYMENT_BURDEN']    = input_df['INSTALL_SIZE'] / (input_df['DUE_PAYMENT'] + 1)
        input_df['IS_HIGH_BALANCE']   = (input_df['ACCCURRENTBALANCE'] > 68348).astype(int)

        prediction  = model.predict(input_df)
        probability = model.predict_proba(input_df)[0][1]

        return jsonify({
            'prediction' : 'DEFAULT' if prediction[0] == 1 else 'NO DEFAULT',
            'probability': round(float(probability) * 100, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("Starting Credit Scoring API...")
    print("Open browser: http://127.0.0.1:5000")
    app.run(debug=True)
