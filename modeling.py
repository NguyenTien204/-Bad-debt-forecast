
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd
from feature_engineering import calculate_risk_metrics
import joblib
# Hàm định dạng báo cáo phân loại thành DataFrame
def format_classification_report(report_str):
    report_data = []
    lines = report_str.split("\n")
    for line in lines[2:len(lines)-3]:  # Bỏ dòng đầu và cuối
        row_data = line.split()
        if len(row_data) != 5:  # Kiểm tra xem dòng có đủ 5 giá trị không
            continue
        try:
            class_label = row_data[0]
            precision = float(row_data[1])
            recall = float(row_data[2])
            f1_score = float(row_data[3])
            support = int(row_data[4])
            report_data.append({
                "Class": class_label,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1_score,
                "Support": support
            })
        except ValueError as e:  # Xử lý lỗi khi chuyển đổi không thành công
            print(f"Không thể xử lý dòng: {line}. Lỗi: {e}")
            continue
    return pd.DataFrame.from_records(report_data)


# Function to evaluate new customers
def evaluate_new_customers(model, new_data, processed_data):
    # Tính toán PD, LGD, EAD, EL
    new_data = calculate_risk_metrics(new_data)
    features = ["PD", "LGD", "EAD", "EL"]

    # Dự đoán rủi ro cho khách hàng mới
    predictions = model.predict(new_data[features])
    probabilities = model.predict_proba(new_data[features])[:, 1]

    new_data["Predicted_Default"] = predictions
    new_data["Default_Probability"] = probabilities

    return new_data

from sklearn.metrics import classification_report, roc_auc_score
import joblib

def train_model(data):
    # Đường dẫn tới file mô hình đã lưu
    reg_model_file = "risk_management/modelfile/multioutput_model.pkl"
    clf_model_file = "risk_management/modelfile/classification_model.pkl"
    
    # Load mô hình
    reg_model = joblib.load(reg_model_file)
    clf_model = joblib.load(clf_model_file)
    
    # Chuẩn bị dữ liệu
    features = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", 
                "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6", 
                "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", 
                "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6", 
                "Credit_Utilization", "Repayment_Trend"]

    X = data[features]
    y_class = data["default"]  
    y_reg = data[["PD", "LGD", "EAD", "EL"]]  
    
    # Chia tập dữ liệu
    X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)
    _, _, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    
    # Đánh giá mô hình phân loại
    y_pred_class = clf_model.predict(X_test)
    y_proba_class = clf_model.predict_proba(X_test)[:, 1]
    classification_report_str = classification_report(y_test_class, y_pred_class)
    auc_roc_class = roc_auc_score(y_test_class, y_proba_class)
    
    # Đánh giá mô hình hồi quy
    y_pred_reg = reg_model.predict(X_test)
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    mse_reg = mean_squared_error(y_test_reg, y_pred_reg)
    r2_reg = r2_score(y_test_reg, y_pred_reg)
    mae_reg = mean_absolute_error(y_test_reg, y_pred_reg)
    
    # Kết quả đánh giá
    classification_results = {
        "classification_report": classification_report_str,
        "auc_roc_class": auc_roc_class
    }
    regression_results = {
        "MSE": mse_reg,
        "R²": r2_reg,
        "MAE": mae_reg
    }
    
    return classification_results, regression_results



