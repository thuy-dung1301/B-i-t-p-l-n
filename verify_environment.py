"""
Verification Script for Fraud-AML Analysis Environment
Kiểm tra xem tất cả các packages đã cài đặt chính xác hay chưa
"""

import sys
from importlib import import_module

def check_package(package_name, display_name=None):
    """Kiểm tra một package có cài đặt không"""
    if display_name is None:
        display_name = package_name
    
    try:
        mod = import_module(package_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✅ {display_name:<25} {version}")
        return True
    except ImportError:
        print(f"❌ {display_name:<25} NOT INSTALLED")
        return False

def main():
    print("=" * 60)
    print("  Fraud-AML Analysis Environment Verification")
    print("=" * 60)
    print()
    
    # Kiểm tra Python version
    print(f"Python Version: {sys.version}")
    print()
    
    # Danh sách các packages cần kiểm tra
    packages = [
        # Core Data Processing
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("openpyxl", "OpenPyXL"),
        
        # Machine Learning
        ("sklearn", "Scikit-learn"),
        ("scipy", "SciPy"),
        ("xgboost", "XGBoost"),
        ("lightgbm", "LightGBM"),
        ("catboost", "CatBoost"),
        
        # Anomaly Detection
        ("pyod", "PyOD"),
        ("imblearn", "Imbalanced-learn"),
        
        # Data Visualization
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("plotly", "Plotly"),
        
        # Statistical Analysis
        ("statsmodels", "StatsModels"),
        
        # Model Explainability
        ("shap", "SHAP"),
        
        # Data Profiling
        ("pandas_profiling", "Pandas Profiling"),
        
        # Jupyter
        ("jupyter", "Jupyter"),
        ("jupyterlab", "JupyterLab"),
        
        # Time Series
        ("tslearn", "TSLearn"),
        ("prophet", "Prophet"),
        
        # NLP
        ("nltk", "NLTK"),
        
        # Database
        ("sqlalchemy", "SQLAlchemy"),
        
        # Utilities
        ("dotenv", "Python-dotenv"),
        ("yaml", "PyYAML"),
        ("loguru", "Loguru"),
    ]
    
    installed_count = 0
    total_count = len(packages)
    
    print("📦 Checking Packages:")
    print("-" * 60)
    
    for package, display_name in packages:
        if check_package(package, display_name):
            installed_count += 1
    
    print("-" * 60)
    print()
    
    # Tóm tắt
    percentage = (installed_count / total_count) * 100
    print(f"📊 Summary: {installed_count}/{total_count} packages installed ({percentage:.1f}%)")
    print()
    
    if installed_count == total_count:
        print("✅ TẤT CẢ PACKAGES ĐÃ CÀI ĐẶT THÀNH CÔNG!")
        print()
        print("🚀 Bạn đã sẵn sàng sử dụng:")
        print("   • Phát hiện giao dịch bất thường (Anomaly Detection)")
        print("   • Phân tích gian lận (Fraud Analysis)")
        print("   • Phân tích AML (Anti-Money Laundering)")
        print("   • Lập báo cáo tài chính (Financial Reporting)")
        return 0
    else:
        missing = total_count - installed_count
        print(f"⚠️  {missing} package(s) chưa được cài đặt")
        print()
        print("💡 Để cài đặt các packages còn thiếu:")
        print("   conda activate fraud-aml-analysis")
        print("   pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())
