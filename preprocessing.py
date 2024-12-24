import pandas as pd

def preprocess_data(data):
    data.rename(columns={"default.payment.next.month": "default"}, inplace=True)
    data.drop(columns=["ID"], inplace=True)
    return data