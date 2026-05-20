"""
Visualization Module - Biểu đồ & Báo cáo Trực quan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Cài đặt style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 8)

class FraudVisualization:
    """Lớp tạo biểu đồ phát hiện gian lận"""
    
    def __init__(self, data):
        """
        Khởi tạo với dữ liệu
        
        Args:
            data: DataFrame chứa kết quả phát hiện
        """
        self.data = data
        
    def plot_risk_distribution(self, save_path=None):
        """Biểu đồ phân bố rủi ro"""
        print("📊 Tạo biểu đồ phân bố rủi ro...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Risk levels
        risk_counts = self.data['Overall_Risk'].value_counts()
        colors = {'CRITICAL': '#d62728', 'HIGH': '#ff7f0e', 'MEDIUM': '#ffbb78', 'LOW': '#2ca02c'}
        color_map = [colors.get(risk, '#1f77b4') for risk in risk_counts.index]
        
        axes[0, 0].bar(risk_counts.index, risk_counts.values, color=color_map)
        axes[0, 0].set_title('Phân bố mức độ rủi ro tổng thể', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Số lượng giao dịch')
        for i, v in enumerate(risk_counts.values):
            axes[0, 0].text(i, v + 5, str(v), ha='center')
        
        # 2. AML Risk distribution
        aml_counts = self.data['AML_Risk'].value_counts()
        axes[0, 1].bar(aml_counts.index, aml_counts.values, color=['#d62728', '#ff7f0e', '#ffbb78', '#2ca02c'][:len(aml_counts)])
        axes[0, 1].set_title('Phân bố AML Risk', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Số lượng giao dịch')
        
        # 3. Fraud Score distribution
        axes[1, 0].hist(self.data['Fraud_Score'], bins=30, color='#1f77b4', edgecolor='black')
        axes[1, 0].set_title('Phân bố Fraud Score', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Fraud Score')
        axes[1, 0].set_ylabel('Tần suất')
        
        # 4. Ensemble Score distribution
        axes[1, 1].hist(self.data['Ensemble_Score'], bins=30, color='#ff7f0e', edgecolor='black')
        axes[1, 1].set_title('Phân bố Ensemble Score', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Ensemble Score')
        axes[1, 1].set_ylabel('Tần suất')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Lưu: {save_path}")
        else:
            plt.show()
        
        return fig
    
    def plot_amount_analysis(self, save_path=None):
        """Phân tích số tiền"""
        print("💰 Tạo biểu đồ phân tích số tiền...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Amount by Risk Level
        self.data.boxplot(column='Amount', by='Overall_Risk', ax=axes[0, 0])
        axes[0, 0].set_title('Số tiền theo mức độ rủi ro')
        axes[0, 0].set_xlabel('Mức độ rủi ro')
        axes[0, 0].set_ylabel('Số tiền')
        
        # 2. Amount by Transaction Type
        self.data.boxplot(column='Amount', by='TransactionType', ax=axes[0, 1])
        axes[0, 1].set_title('Số tiền theo loại giao dịch')
        axes[0, 1].set_xlabel('Loại giao dịch')
        axes[0, 1].set_ylabel('Số tiền')
        
        # 3. Top merchants by fraud score
        merchant_fraud = self.data.groupby('Merchant')['Fraud_Score'].sum().sort_values(ascending=False).head(10)
        axes[1, 0].barh(merchant_fraud.index, merchant_fraud.values, color='#d62728')
        axes[1, 0].set_title('Top 10 Merchant theo Fraud Score')
        axes[1, 0].set_xlabel('Fraud Score')
        
        # 4. Average transaction amount by location
        location_avg = self.data.groupby('Location')['Amount'].mean().sort_values(ascending=False)
        axes[1, 1].barh(location_avg.index, location_avg.values, color='#2ca02c')
        axes[1, 1].set_title('Trung bình số tiền theo Location')
        axes[1, 1].set_xlabel('Số tiền trung bình')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Lưu: {save_path}")
        else:
            plt.show()
        
        return fig
    
    def plot_time_analysis(self, save_path=None):
        """Phân tích theo thời gian"""
        print("⏰ Tạo biểu đồ phân tích theo thời gian...")
        
        if 'Timestamp' not in self.data.columns:
            print("⚠️ Không có cột Timestamp")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Transactions by hour
        hour_counts = self.data['Hour'].value_counts().sort_index()
        axes[0, 0].bar(hour_counts.index, hour_counts.values, color='#1f77b4')
        axes[0, 0].set_title('Số lượng giao dịch theo giờ')
        axes[0, 0].set_xlabel('Giờ trong ngày')
        axes[0, 0].set_ylabel('Số lượng')
        
        # 2. Fraud score by hour
        hour_fraud = self.data.groupby('Hour')['Fraud_Score'].mean()
        axes[0, 1].plot(hour_fraud.index, hour_fraud.values, marker='o', color='#d62728', linewidth=2)
        axes[0, 1].set_title('Fraud Score trung bình theo giờ')
        axes[0, 1].set_xlabel('Giờ trong ngày')
        axes[0, 1].set_ylabel('Fraud Score')
        axes[0, 1].grid(True)
        
        # 3. Transactions by day of week
        day_names = ['Thứ 2', 'Thứ 3', 'Thứ 4', 'Thứ 5', 'Thứ 6', 'Thứ 7', 'Chủ nhật']
        dow_counts = self.data['DayOfWeek'].value_counts().sort_index()
        axes[1, 0].bar([day_names[i] for i in dow_counts.index], dow_counts.values, color='#2ca02c')
        axes[1, 0].set_title('Số lượng giao dịch theo ngày trong tuần')
        axes[1, 0].set_ylabel('Số lượng')
        plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)
        
        # 4. Risk level by hour
        hour_risk = self.data.groupby('Hour')['Overall_Risk'].apply(
            lambda x: (x == 'CRITICAL').sum()
        )
        axes[1, 1].bar(hour_risk.index, hour_risk.values, color='#ff7f0e')
        axes[1, 1].set_title('Số giao dịch CRITICAL theo giờ')
        axes[1, 1].set_xlabel('Giờ trong ngày')
        axes[1, 1].set_ylabel('Số lượng')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Lưu: {save_path}")
        else:
            plt.show()
        
        return fig
    
    def plot_correlation_heatmap(self, save_path=None):
        """Heatmap tương quan"""
        print("🔥 Tạo heatmap tương quan...")
        
        # Chọn các cột numerical
        numeric_cols = ['Amount', 'Fraud_Score', 'AML_Score', 'Ensemble_Score', 
                       'Hour', 'DayOfWeek', 'IF_Score', 'LOF_Score']
        numeric_cols = [col for col in numeric_cols if col in self.data.columns]
        
        correlation = self.data[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap='RdBu_r', center=0, 
                   square=True, ax=ax, cbar_kws={'label': 'Correlation'})
        ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Lưu: {save_path}")
        else:
            plt.show()
        
        return fig
    
    def plot_interactive_scatter(self, save_path=None):
        """Biểu đồ scatter tương tác"""
        print("📍 Tạo biểu đồ scatter tương tác...")
        
        fig = px.scatter(self.data, 
                        x='Amount', 
                        y='Fraud_Score',
                        color='Overall_Risk',
                        size='Ensemble_Score',
                        hover_name='TransactionID',
                        hover_data=['Merchant', 'Location', 'TransactionType'],
                        color_discrete_map={
                            'CRITICAL': '#d62728',
                            'HIGH': '#ff7f0e',
                            'MEDIUM': '#ffbb78',
                            'LOW': '#2ca02c'
                        },
                        title='Fraud Score vs Amount (màu theo Risk Level)',
                        labels={'Amount': 'Số tiền', 'Fraud_Score': 'Fraud Score'})
        
        if save_path:
            fig.write_html(save_path)
            print(f"✅ Lưu: {save_path}")
        else:
            fig.show()
        
        return fig
    
    def plot_interactive_sunburst(self, save_path=None):
        """Biểu đồ Sunburst"""
        print("🌞 Tạo biểu đồ Sunburst...")
        
        # Chuẩn bị dữ liệu
        sunburst_data = self.data.groupby(['Overall_Risk', 'AML_Risk', 'TransactionType']).agg({
            'TransactionID': 'count',
            'Fraud_Score': 'mean'
        }).reset_index()
        sunburst_data.columns = ['Overall_Risk', 'AML_Risk', 'TransactionType', 'Count', 'Fraud_Score']
        
        fig = go.Figure(go.Sunburst(
            labels=sunburst_data['Overall_Risk'].tolist() + 
                   sunburst_data['AML_Risk'].tolist() + 
                   sunburst_data['TransactionType'].tolist(),
            parents=[''] * len(sunburst_data['Overall_Risk'].unique()) +
                   sunburst_data['Overall_Risk'].tolist() +
                   sunburst_data['AML_Risk'].tolist(),
            values=sunburst_data['Count'].tolist(),
            marker=dict(colorscale='RdYlGn_r'),
            title='Phân cấp Risk Levels'
        ))
        
        if save_path:
            fig.write_html(save_path)
            print(f"✅ Lưu: {save_path}")
        else:
            fig.show()
        
        return fig
    
    def generate_all_visualizations(self, output_dir='.'):
        """Tạo tất cả biểu đồ"""
        print(f"\n🎨 Tạo tất cả biểu đồ sang {output_dir}...")
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        self.plot_risk_distribution(f"{output_dir}/01_risk_distribution.png")
        self.plot_amount_analysis(f"{output_dir}/02_amount_analysis.png")
        self.plot_time_analysis(f"{output_dir}/03_time_analysis.png")
        self.plot_correlation_heatmap(f"{output_dir}/04_correlation_heatmap.png")
        self.plot_interactive_scatter(f"{output_dir}/05_interactive_scatter.html")
        self.plot_interactive_sunburst(f"{output_dir}/06_interactive_sunburst.html")
        
        print(f"\n✅ Tất cả biểu đồ đã tạo xong!")


if __name__ == "__main__":
    # Ví dụ sử dụng
    from fraud_detection_aml import FraudDetectionAML
    
    detector = FraudDetectionAML()
    detector.load_data("Bài tập/financial_anomaly_data.csv")
    detector.preprocess_data()
    detector.detect_anomalies_isolation_forest()
    detector.detect_anomalies_lof()
    detector.detect_fraud_patterns()
    detector.detect_aml_violations()
    detector.detect_statistical_anomalies()
    detector.ensemble_detection()
    
    visualizer = FraudVisualization(detector.data)
    visualizer.generate_all_visualizations('visualizations')
