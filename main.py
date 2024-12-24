import pandas as pd
from modeling import train_model
from visualization import plot_customer_segments, generate_summary_report
from feature_engineering import calculate_risk_metrics, add_features, segment_customers
from preprocessing import preprocess_data
import os

# Đường dẫn tuyệt đối đến file data gốc
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, 'UCI_Credit_Card.csv')

PROCESSED_DATA_PATH = "processed_credit_data.csv"
data = pd.read_csv(data_path)

# Tiền xử lý và tính toán các chỉ số
processed_data = preprocess_data(data)
processed_data = add_features(processed_data)
processed_data = segment_customers(processed_data)
processed_data = calculate_risk_metrics(processed_data)
# Lưu dữ liệu đã xử lý
processed_data.to_csv(PROCESSED_DATA_PATH, index=False)
print(f"Dữ liệu đã được xử lý và lưu tại {PROCESSED_DATA_PATH}")

# Phân khúc khách hàng
processed_data = segment_customers(processed_data)
plot_customer_segments(processed_data)

# Dự báo dòng tiền thanh toán


# Huấn luyện và đánh giá mô hình
model = train_model(processed_data)

# Báo cáo tổng hợp
generate_summary_report(processed_data)