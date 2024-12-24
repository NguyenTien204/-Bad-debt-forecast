import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_default_rate_by_category(data, category, top_n=None):
    """
    Biểu đồ tỷ lệ vỡ nợ theo danh mục cụ thể với khả năng lọc top_n danh mục phổ biến nhất.
    """
    if top_n:
        top_categories = data[category].value_counts().head(top_n).index
        data = data[data[category].isin(top_categories)]

    default_rate = data.groupby(category)['default'].mean().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    default_rate.plot(kind='bar', color='salmon', ax=ax)
    ax.set_title(f"Tỷ lệ vỡ nợ theo {category} (Top {top_n if top_n else 'Tất cả'})")
    ax.set_xlabel(category)
    ax.set_ylabel("Tỷ lệ vỡ nợ")
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    
    return fig

def plot_distribution(data, column):
    """
    Biểu đồ phân phối cho một cột cụ thể.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data[column], kde=True, bins=30, color='skyblue', ax=ax)
    ax.set_title(f"Phân phối của {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Tần suất")
    plt.tight_layout()
    return fig

def plot_correlation_heatmap(data, features):
    """
    Biểu đồ heatmap cho các mối tương quan giữa các đặc trưng được chọn.
    """
    corr_matrix = data[features].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Ma trận tương quan")
    plt.tight_layout()
    return fig

def plot_credit_utilization_vs_default(data):
    """
    Scatter plot giữa Credit Utilization và khả năng vỡ nợ.
    """
    fig, ax = plt.subplots(figsize=(8, 2))
    sns.scatterplot(x='Credit_Utilization', y='default', data=data, alpha=0.6, hue='default', palette={0: 'green', 1: 'red'}, ax=ax)
    ax.set_title("Tỷ lệ sử dụng tín dụng so với khả năng vỡ nợ")
    ax.set_xlabel("Tỷ lệ sử dụng tín dụng")
    ax.set_ylabel("Khả năng vỡ nợ")
    plt.tight_layout()
    return fig


def plot_customer_segments_summary(data):
    """
    Biểu đồ phân tích tổng quan các phân khúc khách hàng.
    """
    segment_summary = data.groupby('Customer_Segment')['default'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    segment_summary.plot(kind='bar', color='purple', ax=ax)
    ax.set_title("Tỷ lệ vỡ nợ theo phân khúc khách hàng")
    ax.set_xlabel("Phân khúc khách hàng")
    ax.set_ylabel("Tỷ lệ vỡ nợ")
    ax.tick_params(axis='x', rotation=0)
    plt.tight_layout()
    return fig


#plot_default_rate_by_category(data, 'EDUCATION', top_n=5)
#plot_distribution(data, 'LIMIT_BAL')
#plot_correlation_heatmap(data, ['LIMIT_BAL', 'Credit_Utilization', 'Repayment_Trend', 'PD', 'LGD', 'EAD'])
#plot_credit_utilization_vs_default(data)
#plot_customer_segments_summary(data)

def generate_risk_summary(data):
    summary = {
        "Tổng EL (Expected Loss)": data["EL"].sum(),
        "PD Trung bình": data["PD"].mean(),
        "LGD Trung bình": data["LGD"].mean(),
        "EAD Trung bình": data["EAD"].mean(),
    }
    return summary

def plot_default_by_category(data, category):
    sns.barplot(x=category, y="default", data=data)
    plt.title(f"Tỷ lệ vỡ nợ theo {category}")
    plt.xlabel(category)
    plt.ylabel("Tỷ lệ vỡ nợ")
    plt.show()

def plot_customer_segments(data):
    fig, ax = plt.subplots()
    scatter = ax.scatter(data["LIMIT_BAL"], data["Credit_Utilization"], c=data["Customer_Segment"], cmap="viridis")
    ax.set_xlabel("LIMIT_BAL")
    ax.set_ylabel("Credit Utilization")
    ax.set_title("Phân khúc khách hàng")
    ax.ticklabel_format(style='plain')  
    fig.colorbar(scatter, label="Customer Segment")
    return fig


def generate_summary_report(data):
    print("Báo cáo tổng hợp:")
    print(f"Tổng số khách hàng: {len(data)}")
    print(f"Số lượng khách hàng vỡ nợ: {data['default'].sum()}")
    print(f"Tỷ lệ khách hàng vỡ nợ: {data['default'].mean() * 100:.2f}%")
    print(f"Phân khúc khách hàng:")
    print(data["Customer_Segment"].value_counts())
    print("-" * 30)
    
def plot_risk_heatmap(data):
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(10, 6))
    corr = data[["PD", "LGD", "EAD", "EL"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Tương quan giữa các chỉ số rủi ro")
    return fig
