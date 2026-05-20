"""
Advanced Machine Learning Module - Phát hiện Gian lận Nâng cao
Sử dụng: Random Forest, Gradient Boosting, Neural Networks
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, auc, precision_recall_curve)
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow không được cài đặt. Bỏ qua Deep Learning models.")


class AdvancedFraudDetection:
    """Lớp phát hiện gian lận nâng cao sử dụng Machine Learning"""
    
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.rf_model = None
        self.gb_model = None
        self.nn_model = None
        self.feature_names = None
        
    def load_and_prepare_data(self, data):
        """Tải và chuẩn bị dữ liệu"""
        print("📊 Chuẩn bị dữ liệu cho ML models...")
        
        self.data = data.copy()
        
        # Tạo target variable từ các phương pháp phát hiện khác nhau
        # Giao dịch được coi là gian lận nếu: Ensemble_Score > 5 hoặc Overall_Risk = CRITICAL/HIGH
        self.data['Target'] = 0
        self.data.loc[(self.data['Ensemble_Score'] >= 5) | 
                     (self.data['Overall_Risk'].isin(['CRITICAL', 'HIGH'])), 'Target'] = 1
        
        # Chọn features
        feature_cols = ['Amount', 'Hour', 'DayOfWeek', 'IsPeakHour',
                       'Merchant_encoded', 'TransactionType_encoded', 'Location_encoded']
        feature_cols = [col for col in feature_cols if col in self.data.columns]
        
        self.feature_names = feature_cols
        X = self.data[feature_cols].copy()
        y = self.data['Target'].copy()
        
        # Xử lý missing values
        X = X.fillna(X.mean())
        
        # Chia train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Thống kê
        fraud_rate_train = (self.y_train == 1).sum() / len(self.y_train) * 100
        fraud_rate_test = (self.y_test == 1).sum() / len(self.y_test) * 100
        
        print(f"✅ Training set: {len(self.X_train)} samples ({fraud_rate_train:.2f}% fraud)")
        print(f"✅ Test set: {len(self.X_test)} samples ({fraud_rate_test:.2f}% fraud)")
        print(f"📋 Features: {', '.join(feature_cols)}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_random_forest(self):
        """Huấn luyện Random Forest"""
        print("\n🌲 Huấn luyện Random Forest...")
        
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        self.rf_model.fit(self.X_train_scaled, self.y_train)
        
        # Đánh giá
        train_score = self.rf_model.score(self.X_train_scaled, self.y_train)
        test_score = self.rf_model.score(self.X_test_scaled, self.y_test)
        
        y_pred = self.rf_model.predict(self.X_test_scaled)
        y_pred_proba = self.rf_model.predict_proba(self.X_test_scaled)[:, 1]
        
        roc_score = roc_auc_score(self.y_test, y_pred_proba)
        
        print(f"✅ Train Accuracy: {train_score:.4f}")
        print(f"✅ Test Accuracy: {test_score:.4f}")
        print(f"✅ ROC-AUC Score: {roc_score:.4f}")
        
        self._print_classification_report(y_pred, "Random Forest")
        
        return self.rf_model
    
    def train_gradient_boosting(self):
        """Huấn luyện Gradient Boosting"""
        print("\n📈 Huấn luyện Gradient Boosting...")
        
        self.gb_model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42
        )
        
        self.gb_model.fit(self.X_train_scaled, self.y_train)
        
        # Đánh giá
        train_score = self.gb_model.score(self.X_train_scaled, self.y_train)
        test_score = self.gb_model.score(self.X_test_scaled, self.y_test)
        
        y_pred = self.gb_model.predict(self.X_test_scaled)
        y_pred_proba = self.gb_model.predict_proba(self.X_test_scaled)[:, 1]
        
        roc_score = roc_auc_score(self.y_test, y_pred_proba)
        
        print(f"✅ Train Accuracy: {train_score:.4f}")
        print(f"✅ Test Accuracy: {test_score:.4f}")
        print(f"✅ ROC-AUC Score: {roc_score:.4f}")
        
        self._print_classification_report(y_pred, "Gradient Boosting")
        
        return self.gb_model
    
    def train_neural_network(self):
        """Huấn luyện Neural Network"""
        if not TENSORFLOW_AVAILABLE:
            print("⚠️ TensorFlow không khả dụng, bỏ qua Neural Network")
            return None
        
        print("\n🧠 Huấn luyện Neural Network...")
        
        # Xây dựng mô hình
        self.nn_model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(self.X_train_scaled.shape[1],)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(16, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(1, activation='sigmoid')
        ])
        
        self.nn_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC()]
        )
        
        # Huấn luyện
        history = self.nn_model.fit(
            self.X_train_scaled, self.y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            ]
        )
        
        # Đánh giá
        train_loss, train_acc, train_auc = self.nn_model.evaluate(self.X_train_scaled, self.y_train, verbose=0)
        test_loss, test_acc, test_auc = self.nn_model.evaluate(self.X_test_scaled, self.y_test, verbose=0)
        
        print(f"✅ Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
        print(f"✅ Train AUC: {train_auc:.4f}, Test AUC: {test_auc:.4f}")
        
        return self.nn_model
    
    def _print_classification_report(self, y_pred, model_name):
        """In classification report"""
        print(f"\n{model_name} - Classification Report:")
        print(classification_report(self.y_test, y_pred, 
                                   target_names=['Normal', 'Fraud'],
                                   digits=4))
    
    def get_feature_importance(self):
        """Lấy feature importance từ Random Forest"""
        print("\n📊 Feature Importance:")
        
        if self.rf_model is None:
            print("⚠️ Random Forest chưa được huấn luyện")
            return None
        
        importances = self.rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        importance_df = pd.DataFrame({
            'Feature': [self.feature_names[i] for i in indices],
            'Importance': importances[indices]
        })
        
        print(importance_df.to_string(index=False))
        
        return importance_df
    
    def predict_fraud_probability(self, new_data):
        """Dự đoán xác suất gian lận"""
        if self.rf_model is None:
            print("⚠️ Mô hình chưa được huấn luyện")
            return None
        
        new_data_scaled = self.scaler.transform(new_data)
        probabilities = self.rf_model.predict_proba(new_data_scaled)[:, 1]
        
        return probabilities
    
    def evaluate_all_models(self):
        """Đánh giá tất cả mô hình"""
        print("\n" + "="*60)
        print("📊 EVALUATION SUMMARY - TẤT CẢ MÔ HÌNH")
        print("="*60)
        
        models = {
            'Random Forest': (self.rf_model, 'RF'),
            'Gradient Boosting': (self.gb_model, 'GB'),
            'Neural Network': (self.nn_model, 'NN')
        }
        
        results = {}
        
        for model_name, (model, abbrev) in models.items():
            if model is None:
                continue
            
            print(f"\n{model_name}:")
            
            try:
                if model_name == 'Neural Network' and TENSORFLOW_AVAILABLE:
                    y_pred_proba = model.predict(self.X_test_scaled, verbose=0).ravel()
                else:
                    y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
                
                roc_score = roc_auc_score(self.y_test, y_pred_proba)
                y_pred = (y_pred_proba >= 0.5).astype(int)
                acc_score = (y_pred == self.y_test).mean()
                
                results[model_name] = {
                    'ROC-AUC': roc_score,
                    'Accuracy': acc_score
                }
                
                print(f"  ROC-AUC: {roc_score:.4f}")
                print(f"  Accuracy: {acc_score:.4f}")
            except Exception as e:
                print(f"  ⚠️ Lỗi: {str(e)}")
        
        print("\n" + "="*60)
        return results
    
    def cross_validate_model(self, model=None, cv=5):
        """Cross-validation"""
        if model is None:
            model = self.rf_model
        
        if model is None:
            print("⚠️ Không có mô hình để validate")
            return None
        
        print(f"\n🔄 Cross-Validation (cv={cv})...")
        
        scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                cv=cv, scoring='roc_auc')
        
        print(f"✅ ROC-AUC Scores: {scores}")
        print(f"✅ Mean: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        return scores


class HypothesisTesting:
    """Lớp kiểm định giả thuyết thống kê"""
    
    def __init__(self, data):
        self.data = data
    
    def test_fraud_amount_difference(self):
        """Kiểm định: Giao dịch gian lận có số tiền khác nhau không?"""
        from scipy import stats
        
        print("\n📊 Hypothesis Test: Amount difference between fraud and normal transactions")
        
        fraud = self.data[self.data['Overall_Risk'].isin(['CRITICAL', 'HIGH'])]['Amount']
        normal = self.data[~self.data['Overall_Risk'].isin(['CRITICAL', 'HIGH'])]['Amount']
        
        # T-test
        t_stat, p_value = stats.ttest_ind(fraud, normal)
        
        print(f"Fraud transactions - Mean: {fraud.mean():.2f}, Std: {fraud.std():.2f}")
        print(f"Normal transactions - Mean: {normal.mean():.2f}, Std: {normal.std():.2f}")
        print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.6f}")
        
        if p_value < 0.05:
            print("✅ Kết luận: Có sự khác biệt có ý nghĩa thống kê (p < 0.05)")
        else:
            print("⚠️ Kết luận: Không có sự khác biệt đáng kể")
        
        return t_stat, p_value
    
    def test_location_fraud_correlation(self):
        """Kiểm định: Location có liên quan đến gian lận không?"""
        from scipy.stats import chi2_contingency
        
        print("\n📊 Hypothesis Test: Location correlation with fraud")
        
        contingency_table = pd.crosstab(
            self.data['Location'],
            self.data['Overall_Risk'].isin(['CRITICAL', 'HIGH'])
        )
        
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        print(f"Chi-square: {chi2:.4f}, P-value: {p_value:.6f}")
        
        if p_value < 0.05:
            print("✅ Kết luận: Location có liên quan có ý nghĩa với gian lận")
        else:
            print("⚠️ Kết luận: Không tìm thấy mối liên hệ")
        
        return chi2, p_value


def main():
    """Hàm chính"""
    from fraud_detection_aml import FraudDetectionAML
    
    # Khởi tạo và chạy phát hiện cơ bản
    detector = FraudDetectionAML()
    detector.load_data("Bài tập/financial_anomaly_data.csv")
    detector.preprocess_data()
    detector.detect_anomalies_isolation_forest()
    detector.detect_anomalies_lof()
    detector.detect_fraud_patterns()
    detector.detect_aml_violations()
    detector.detect_statistical_anomalies()
    detector.ensemble_detection()
    
    # Huấn luyện các mô hình nâng cao
    ml_detector = AdvancedFraudDetection()
    ml_detector.load_and_prepare_data(detector.data)
    ml_detector.train_random_forest()
    ml_detector.train_gradient_boosting()
    if TENSORFLOW_AVAILABLE:
        ml_detector.train_neural_network()
    
    # Đánh giá mô hình
    ml_detector.evaluate_all_models()
    
    # Feature importance
    ml_detector.get_feature_importance()
    
    # Cross-validation
    ml_detector.cross_validate_model()
    
    # Kiểm định giả thuyết
    hypothesis = HypothesisTesting(detector.data)
    hypothesis.test_fraud_amount_difference()
    hypothesis.test_location_fraud_correlation()


if __name__ == "__main__":
    main()
