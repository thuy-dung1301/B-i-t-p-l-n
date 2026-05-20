# HƯỚNG DẪN THỰC HÀNH TẠO MÔI TRƯỜNG ẢO BẰNG ANACONDA

## 🎯 MỤC TIÊU
Học cách tạo và sử dụng môi trường ảo (virtual environment) với Anaconda để quản lý các dự án Python độc lập.

## 📋 CÁC BƯỚC THỰC HIỆN

### 1. KIỂM TRA ANACONDA ĐÃ CÀI ĐẶT
```bash
# Kiểm tra phiên bản conda
conda --version

# Hoặc sử dụng đường dẫn đầy đủ
C:\Users\[Tên_User]\miniconda3\Scripts\conda.exe --version
```

### 2. XEM DANH SÁCH MÔI TRƯỜNG HIỆN CÓ
```bash
# Liệt kê tất cả môi trường
conda env list

# Hoặc
C:\Users\[Tên_User]\miniconda3\Scripts\conda.exe env list
```

### 3. TẠO MÔI TRƯỜNG ẢO MỚI
```bash
# Tạo môi trường với Python 3.10
conda create -n data_analysis_env python=3.10 -y

# Tạo môi trường với các package cụ thể
conda create -n my_project python=3.9 pandas numpy matplotlib -y
```

### 4. KÍCH HOẠT MÔI TRƯỜNG ẢO
```bash
# Kích hoạt môi trường
conda activate data_analysis_env

# Kiểm tra môi trường hiện tại
conda info --envs
```

### 5. CÀI ĐẶT CÁC PACKAGE CẦN THIẾT
```bash
# Cài đặt trong môi trường hiện tại
conda install pandas matplotlib seaborn numpy scikit-learn -y

# Cài đặt trong môi trường cụ thể (không cần activate)
conda install -n data_analysis_env jupyter notebook -y

# Cập nhật package
conda update pandas
```

### 6. SỬ DỤNG MÔI TRƯỜNG ẢO
```bash
# Chạy Python script trong môi trường
conda run -n data_analysis_env python script.py

# Mở Jupyter Notebook
conda run -n data_analysis_env jupyter notebook

# Cài đặt package từ requirements.txt
conda run -n data_analysis_env pip install -r requirements.txt
```

### 6.1. CHẠY ỨNG DỤNG WEB PHÁT HIỆN GIAO DỊCH BẤT THƯỜNG
```bash
# Kích hoạt môi trường
conda activate fraud-aml-analysis

# Cài đặt Streamlit nếu chưa có
pip install streamlit

# Chạy ứng dụng web
streamlit run ledger_anomaly_web.py
```

### 7. QUẢN LÝ MÔI TRƯỜNG
```bash
# Thoát môi trường
conda deactivate

# Xóa môi trường
conda env remove -n data_analysis_env

# Xuất danh sách package
conda list -n data_analysis_env > requirements.txt

# Tạo môi trường từ file environment.yml
conda env create -f environment.yml
```

## 🔧 THỰC HÀNH CỤ THỂ

### Tạo môi trường cho phân tích dữ liệu:
```bash
# 1. Tạo môi trường
conda create -n data_analysis python=3.10 -y

# 2. Cài đặt các thư viện cần thiết
conda install -n data_analysis pandas matplotlib seaborn numpy scikit-learn jupyter -y

# 3. Kiểm tra môi trường
conda run -n data_analysis python -c "import pandas as pd; print('Pandas version:', pd.__version__)"
```

### File environment.yml mẫu:
```yaml
name: data_analysis
channels:
  - defaults
dependencies:
  - python=3.10
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - jupyter
  - pip
  - pip:
    - openpyxl
    - xlrd
```

## ✅ KẾT QUẢ THỰC HIỆN

### Môi trường đã tạo thành công:
- **Tên**: `data_analysis_env`
- **Python**: 3.10.20
- **Vị trí**: `C:\Users\[Tên_User]\miniconda3\envs\data_analysis_env`

### Các package đã cài đặt:
- ✅ pandas (2.3.3)
- ✅ numpy (2.2.5)
- ✅ matplotlib (3.10.9)
- ✅ seaborn (0.13.2)

### File kiểm tra:
- `test_environment.py` - Script kiểm tra môi trường
- `env_test_chart.png` - Biểu đồ kiểm tra

## 🎯 LỢI ÍCH CỦA MÔI TRƯỜNG ẢO

1. **Tách biệt dự án**: Mỗi dự án có môi trường riêng
2. **Quản lý phiên bản**: Kiểm soát version của Python và packages
3. **Tránh xung đột**: Không bị ảnh hưởng bởi các dự án khác
4. **Dễ chia sẻ**: Xuất environment.yml để đồng đội sử dụng
5. **Dễ bảo trì**: Có thể xóa và tạo lại môi trường dễ dàng

## 🚀 MẸO SỬ DỤNG

1. **Luôn sử dụng môi trường ảo** cho mọi dự án mới
2. **Xuất environment.yml** để chia sẻ dự án
3. **Cập nhật conda** định kỳ: `conda update -n base conda`
4. **Dọn dẹp cache**: `conda clean --all`
5. **Sử dụng conda run** thay vì activate khi automation

## 📚 TÀI LIỆU THAM KHẢO

- [Conda Documentation](https://docs.conda.io/)
- [Managing Environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
- [Conda Cheat Sheet](https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html)

---

*Tạo bởi: AI Assistant*
*Ngày: 11/05/2026*