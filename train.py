import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from feature_engineering import add_features,segment_customers,calculate_risk_metrics
from preprocessing import preprocess_data
from imblearn.over_sampling import SMOTE
# Đọc dữ liệu từ file CSV
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, 'processed_credit_data.csv')

data = pd.read_csv(data_path)
data = add_features(data)
data = segment_customers(data)
data = calculate_risk_metrics(data)
# Chuẩn bị dữ liệu
predic_features = ["LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6","BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6","Credit_Utilization", "Repayment_Trend"]
X = data[predic_features]
predic_targets = ["PD", "LGD", "EAD", "EL"]
features_targets = ["default"]
y_predic = data[predic_targets]
y_targets = data[features_targets]
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y_targets)

# Chia dữ liệu huấn luyện và kiểm tra
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_predic, test_size=0.2, random_state=42)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)
# Huấn luyện mô hình RandomForest với đa đầu ra
reg_model = RandomForestRegressor(random_state=42)
reg_model.fit(X_train_reg, y_train_reg)

# Huấn luyện mô hình phân loại
clf_model = RandomForestClassifier(random_state=42)
clf_model.fit(X_train_clf, y_train_clf)

# Lưu mô hình
predic_model_file = "multioutput_model.pkl"
clf_model_file = "classification_model.pkl"
joblib.dump(reg_model, predic_model_file)
joblib.dump(clf_model, clf_model_file)

print(f"Mô hình hồi quy đã được lưu tại {predic_model_file}")
print(f"Mô hình phân loại đã được lưu tại {clf_model_file}")