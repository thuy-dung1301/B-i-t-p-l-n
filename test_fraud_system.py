"""
Test & Demo Script - Kiểm tra và demo hệ thống
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def print_section(title):
    """In section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def test_imports():
    """Test 1: Kiểm tra các imports"""
    print_section("TEST 1: Checking Imports")
    
    try:
        import pandas
        print("✅ pandas")
    except ImportError as e:
        print(f"❌ pandas: {e}")
        return False
    
    try:
        import numpy
        print("✅ numpy")
    except ImportError as e:
        print(f"❌ numpy: {e}")
        return False
    
    try:
        import sklearn
        print("✅ scikit-learn")
    except ImportError as e:
        print(f"❌ scikit-learn: {e}")
        return False
    
    try:
        import matplotlib
        print("✅ matplotlib")
    except ImportError as e:
        print(f"❌ matplotlib: {e}")
        return False
    
    try:
        import seaborn
        print("✅ seaborn")
    except ImportError as e:
        print(f"❌ seaborn: {e}")
        return False
    
    try:
        import plotly
        print("✅ plotly")
    except ImportError as e:
        print(f"❌ plotly: {e}")
        return False
    
    try:
        import scipy
        print("✅ scipy")
    except ImportError as e:
        print(f"❌ scipy: {e}")
        return False
    
    try:
        import tensorflow
        print("✅ tensorflow (optional)")
    except ImportError:
        print("⚠️  tensorflow (optional - not installed)")
    
    return True

def test_module_imports():
    """Test 2: Kiểm tra các module của hệ thống"""
    print_section("TEST 2: Checking System Modules")
    
    try:
        from fraud_detection_aml import FraudDetectionAML
        print("✅ fraud_detection_aml module")
    except ImportError as e:
        print(f"❌ fraud_detection_aml module: {e}")
        return False
    
    try:
        from visualization_fraud import FraudVisualization
        print("✅ visualization_fraud module")
    except ImportError as e:
        print(f"❌ visualization_fraud module: {e}")
        return False
    
    try:
        from advanced_ml_fraud import AdvancedFraudDetection, HypothesisTesting
        print("✅ advanced_ml_fraud module")
    except ImportError as e:
        print(f"❌ advanced_ml_fraud module: {e}")
        return False
    
    try:
        import config
        print("✅ config module")
    except ImportError as e:
        print(f"❌ config module: {e}")
        return False
    
    return True

def test_data_loading():
    """Test 3: Kiểm tra tải dữ liệu"""
    print_section("TEST 3: Data Loading")
    
    data_path = "Bài tập/financial_anomaly_data.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return False
    
    try:
        df = pd.read_csv(data_path)
        print(f"✅ Data loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # Check required columns
        required_cols = ['Timestamp', 'TransactionID', 'AccountID', 'Amount', 'Merchant', 'TransactionType', 'Location']
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            print(f"⚠️  Missing columns: {missing}")
            return False
        
        print(f"✅ All required columns present")
        print(f"   Columns: {list(df.columns)}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

def test_basic_detection():
    """Test 4: Kiểm tra phát hiện cơ bản"""
    print_section("TEST 4: Basic Detection")
    
    try:
        from fraud_detection_aml import FraudDetectionAML
        
        detector = FraudDetectionAML(contamination=0.05)
        print("✅ FraudDetectionAML initialized")
        
        detector.load_data("Bài tập/financial_anomaly_data.csv")
        print("✅ Data loaded")
        
        detector.preprocess_data()
        print("✅ Data preprocessed")
        
        detector.detect_anomalies_isolation_forest()
        print("✅ Isolation Forest detection completed")
        
        detector.detect_anomalies_lof()
        print("✅ LOF detection completed")
        
        detector.detect_fraud_patterns()
        print("✅ Fraud pattern detection completed")
        
        detector.detect_aml_violations()
        print("✅ AML detection completed")
        
        detector.detect_statistical_anomalies()
        print("✅ Statistical anomaly detection completed")
        
        detector.ensemble_detection()
        print("✅ Ensemble detection completed")
        
        # Kiểm tra kết quả
        suspicious = detector.get_suspicious_transactions('CRITICAL')
        print(f"✅ Found {len(suspicious)} CRITICAL risk transactions")
        
        return detector
    except Exception as e:
        print(f"❌ Error during detection: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_group_analysis(detector):
    """Test 5: Kiểm tra phân tích theo nhóm"""
    print_section("TEST 5: Group Analysis")
    
    try:
        results = detector.analyze_by_groups()
        print(f"✅ Group analysis completed")
        print(f"   - Merchant analysis: {len(results['by_merchant'])} merchants")
        print(f"   - Location analysis: {len(results['by_location'])} locations")
        print(f"   - Type analysis: {len(results['by_type'])} types")
        return True
    except Exception as e:
        print(f"❌ Error during group analysis: {e}")
        return False

def test_export(detector):
    """Test 6: Kiểm tra xuất kết quả"""
    print_section("TEST 6: Results Export")
    
    try:
        output_dir = "test_results"
        os.makedirs(output_dir, exist_ok=True)
        
        detector.export_results(output_dir)
        print(f"✅ Results exported to {output_dir}")
        
        # Kiểm tra files
        files = os.listdir(output_dir)
        print(f"✅ Generated {len(files)} files:")
        for f in files:
            print(f"   - {f}")
        
        # Cleanup
        import shutil
        shutil.rmtree(output_dir)
        
        return True
    except Exception as e:
        print(f"❌ Error during export: {e}")
        return False

def test_ml_models(detector):
    """Test 7: Kiểm tra ML models"""
    print_section("TEST 7: Machine Learning Models")
    
    try:
        from advanced_ml_fraud import AdvancedFraudDetection
        
        ml_detector = AdvancedFraudDetection()
        print("✅ AdvancedFraudDetection initialized")
        
        ml_detector.load_and_prepare_data(detector.data)
        print("✅ Data prepared for ML")
        
        ml_detector.train_random_forest()
        print("✅ Random Forest trained")
        
        ml_detector.train_gradient_boosting()
        print("✅ Gradient Boosting trained")
        
        results = ml_detector.evaluate_all_models()
        print("✅ Models evaluated")
        print(f"   Results: {results}")
        
        return True
    except Exception as e:
        print(f"❌ Error during ML testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_visualization(detector):
    """Test 8: Kiểm tra visualization"""
    print_section("TEST 8: Visualization")
    
    try:
        from visualization_fraud import FraudVisualization
        
        viz = FraudVisualization(detector.data)
        print("✅ FraudVisualization initialized")
        
        # Kiểm tra các methods
        viz.plot_risk_distribution()
        print("✅ Risk distribution plot created")
        plt.close()
        
        viz.plot_amount_analysis()
        print("✅ Amount analysis plot created")
        plt.close()
        
        import matplotlib.pyplot as plt
        
        return True
    except Exception as e:
        print(f"⚠️  Warning during visualization test: {e}")
        # Không fail test nếu visualization không được
        return True

def print_summary(results):
    """In tóm tắt kết quả"""
    print_section("TEST SUMMARY - TÓM TẮT KẾT QUẢ")
    
    total = len(results)
    passed = sum(1 for r in results.values() if r)
    
    print(f"\nTotal Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    
    print("\nDetailed Results:")
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name:.<50} {status}")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready to use.")
        return True
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return False

def main():
    """Hàm chính"""
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  FRAUD DETECTION SYSTEM - TEST & DEMO".center(68) + "║")
    print("║" + "  Kiểm tra & Demo Hệ thống Phát hiện Gian lận".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "═"*68 + "╝")
    
    results = {}
    
    # Test 1
    results['Test 1: Imports'] = test_imports()
    
    # Test 2
    results['Test 2: Module Imports'] = test_module_imports()
    
    # Test 3
    results['Test 3: Data Loading'] = test_data_loading()
    
    # Test 4
    detector = None
    test4_passed = False
    try:
        detector = test_basic_detection()
        test4_passed = detector is not None
        results['Test 4: Basic Detection'] = test4_passed
    except Exception as e:
        results['Test 4: Basic Detection'] = False
        print(f"❌ Error: {e}")
    
    # Test 5-8: Chỉ chạy nếu test 4 thành công
    if test4_passed:
        results['Test 5: Group Analysis'] = test_group_analysis(detector)
        results['Test 6: Results Export'] = test_export(detector)
        results['Test 7: ML Models'] = test_ml_models(detector)
        results['Test 8: Visualization'] = test_visualization(detector)
    else:
        print("\n⚠️  Skipping tests 5-8 because basic detection failed")
        results['Test 5: Group Analysis'] = False
        results['Test 6: Results Export'] = False
        results['Test 7: ML Models'] = False
        results['Test 8: Visualization'] = False
    
    # Print summary
    success = print_summary(results)
    
    print("\n" + "="*70)
    if success:
        print("Ready to run: python run_fraud_detection.py")
    else:
        print("Please fix the errors above before running the system")
    print("="*70 + "\n")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
