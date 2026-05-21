import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# 1. Cấu hình trang giao diện Web
st.set_page_config(page_title="Hệ thống Phát hiện Gian lận", layout="wide")
st.title("🚨 ỨNG DỤNG PHÁT HIỆN GIAO DỊCH GIAN LẬN")
st.write("Ứng dụng sử dụng thuật toán thông minh **Isolation Forest** để tự động cô lập và tìm ra các giao dịch bất thường.")

# 2. Tạo khu vực tải file ở thanh bên cạnh (Sidebar)
st.sidebar.header("Dữ liệu đầu vào")
uploaded_file = st.sidebar.file_uploader("Tải file dữ liệu giao dịch (.csv) tại đây", type=["csv"])

# 3. Hàm xử lý thuật toán Isolation Forest
def run_isolation_forest(data):
    # Sao chép dữ liệu tránh làm ảnh hưởng file gốc
    df = data.copy()
    
    # Giả định tìm các cột số để đưa vào tính toán (Số tiền, Thời gian...)
    # Hệ thống sẽ tự động lấy các cột chứa số để phân tích đám đông
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) == 0:
        st.error("File của bạn không có cột dữ liệu dạng số nào để phân tích!")
        return None

    # Chuẩn hóa dữ liệu về cùng một thang đo
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numeric_cols].fillna(0))
    
    # Khởi tạo thuật toán Isolation Forest
    # contamination=0.05 nghĩa là chúng ta đoán có khoảng 5% giao dịch trong file là gian lận
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(scaled_data)

    df['Anomaly_Score'] = model.decision_function(scaled_data)
    predictions = model.predict(scaled_data)
    
    # Thuật toán trả về: -1 là Bất thường (Gian lận), 1 là Bình thường
    df['Trạng Thái'] = np.where(predictions == -1, '🚨 Gian lận / Bất thường', '✅ Bình thường')
    
    return df

# 4. Luồng xử lý hiển thị trên giao diện Web
if uploaded_file is not None:
    # Đọc file dữ liệu người dùng tải lên
    df_raw = pd.read_csv(uploaded_file)
    
    st.subheader("📊 Xem trước dữ liệu vừa tải lên")
    st.dataframe(df_raw.head(10)) # Hiển thị 10 dòng đầu tiên
    
    # Bấm nút để kích hoạt AI chạy
    if st.button("Chạy thuật toán kiểm tra gian lận"):
        with st.spinner('AI đang tiến hành phân tích và cô lập dữ liệu...'):
            result_df = run_isolation_forest(df_raw)
            
            if result_df is not None:
                st.success("🎉 Đã phân tích xong!")
                
                # Thống kê kết quả
                total = len(result_df)
                fraud_count = len(result_df[result_df['Trạng Thái'] == '🚨 Gian lận / Bất thường'])
                
                # Hiển thị số liệu tổng quan dạng thẻ (Metrics)
                col1, col2 = st.columns(2)
                col1.metric("Tổng số giao dịch", f"{total} dòng")
                col2.metric("Phát hiện nghi vấn", f"{fraud_count} dòng", delta=f"{fraud_count/total*100:.1f}%", delta_color="inverse")
                
                # Hiển thị danh sách kết quả chi tiết
                st.subheader("🔍 Danh sách kết quả chi tiết")
                st.dataframe(result_df)
                
                # Cho phép người dùng tải file kết quả đã lọc về máy
                csv_ready = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Tải file kết quả phân tích về máy (.CSV)",
                    data=csv_ready,
                    file_name='ket_qua_gian_lan.csv',
                    mime='text/csv',
                )
else:
    # Giao diện khi chưa tải file lên
    st.info("💡 Vui lòng tải một file dữ liệu dạng `.csv` ở thanh bên trái để hệ thống bắt đầu phân tích nhé.")
    
    # Tạo sẵn một nút bấm để người dùng test thử bằng dữ liệu giả lập nếu chưa có file
    if st.button("Chạy thử bằng dữ liệu giả lập của hệ thống"):
        np.random.seed(42)
        # Tạo 100 dòng bình thường
        normal_data = np.random.normal(loc=10, scale=2, size=(100, 2))
        # Tạo 5 dòng bất thường (số tiền cực to, thời gian kỳ lạ)
        anomaly_data = np.random.uniform(low=50, high=100, size=(5, 2))
        
        all_data = np.vstack([normal_data, anomaly_data])
        df_demo = pd.DataFrame(all_data, columns=['Số tiền (Triệu)', 'Thời gian giao dịch (Phút)'])
        
        st.subheader("📊 Dữ liệu mẫu giả lập:")
        st.dataframe(df_demo)
        
        # Chạy thuật toán trên dữ liệu giả lập
        res = run_isolation_forest(df_demo)
        st.subheader("🔍 Kết quả chạy thử:")
        st.dataframe(res)