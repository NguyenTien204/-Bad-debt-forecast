from sklearn.cluster import KMeans



def calculate_risk_metrics(data):
    # Tính Probability of Default (PD)
    data["PD"] = data["Repayment_Trend"] / (data["Repayment_Trend"].max() + 1e-5)
    
    # Tính Loss Given Default (LGD)
    data["LGD"] = 1 - data["Credit_Utilization"]
    
    # Tính Exposure at Default (EAD)
    data["EAD"] = data["LIMIT_BAL"]
    
    # Tính Expected Loss (EL)
    data["EL"] = data["PD"] * data["LGD"] * data["EAD"]
    
    return data

def add_features(data):
    # Tính tỷ lệ nợ trên thu nhập (DTI)
    data["DTI"] = data["LIMIT_BAL"] / (data["PAY_AMT1"] + 1e-5)
    
    # Tỷ lệ sử dụng tín dụng
    data["Credit_Utilization"] = (data[["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", 
                                        "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]].mean(axis=1)) / data["LIMIT_BAL"]
    
    # Xu hướng trả nợ (tổng số lần trả chậm trong 6 tháng)
    data["Repayment_Trend"] = data[["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]].clip(lower=0).sum(axis=1)

    
    return data

def segment_customers(data, n_clusters=4):
    features = data[["LIMIT_BAL", "Credit_Utilization", "Repayment_Trend", "AGE"]]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data["Customer_Segment"] = kmeans.fit_predict(features)
    return data
