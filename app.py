from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model và scaler
with open('best_logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    prediction_text = ""
    return render_template('index.html', prediction_text=prediction_text)

@app.route('/predict', methods=['POST'])
def predict():
    # Lấy dữ liệu đầu vào từ form
    age = int(request.form.get('age'))
    sex = int(request.form.get('sex'))
    race = int(request.form.get('race'))
    driving_experience = int(request.form.get('driving_experience'))
    education = int(request.form.get('education'))
    income = int(request.form.get('income'))
    credit_score = float(request.form.get('CREDIT_SCORE'))
    vehicle_ownership = int(request.form.get('vehicle_ownership'))
    vehicle_year = int(request.form.get('vehicle_year'))
    married = int(request.form.get('married'))
    children = int(request.form.get('children'))
    postal_code = int(request.form.get('postal_code'))
    annual_mileage = float(request.form.get('attr13'))
    vehicle_type = int(request.form.get('vehicle_type'))
    speeding_violations = int(request.form.get('SPEEDING_VIOLATIONS'))
    duis = int(request.form.get('DUIS'))
    past_accidents = int(request.form.get('PAST_ACCIDENTS'))

    # Chuyển đổi dữ liệu đầu vào thành array
    input_values = [age, sex, race, driving_experience, education, income, credit_score, 
                    vehicle_ownership, vehicle_year, married, children, postal_code, 
                    annual_mileage, vehicle_type, speeding_violations, duis, past_accidents]
    input_features = np.array([input_values])

    # Chuẩn hóa dữ liệu với scaler đã huấn luyện
    input_data_normalized = scaler.transform(input_features)
    
    # Thực hiện dự đoán
    prediction = model.predict(input_data_normalized)

    # Hiển thị kết quả
   # Hiển thị kết quả dựa trên giá trị dự đoán
    if prediction[0] == 0:
        prediction_text = 'Khách hàng không mua bảo hiểm xe.'
    else:
        prediction_text = 'Khách hàng mua bảo hiểm xe.'
        
    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
