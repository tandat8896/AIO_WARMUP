# CSV Data Explorer with PyGWalker

Ứng dụng Streamlit cho phép người dùng tải lên file CSV và thực hiện Exploratory Data Analysis (EDA) với PyGWalker.

## Tính năng

- Tải lên và đọc dữ liệu từ file CSV
- Xem trước dữ liệu (data preview)
- Hiển thị thông tin thống kê của dữ liệu
- Khám phá dữ liệu tương tác với giao diện kéo thả của PyGWalker
- Tạo biểu đồ và trực quan hóa dữ liệu dễ dàng

## Cài đặt

1. Clone repository này về máy của bạn
2. Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

## Cách sử dụng

1. Chạy ứng dụng Streamlit:

```bash
streamlit run app.py
```

2. Truy cập ứng dụng qua trình duyệt web tại địa chỉ được hiển thị trong terminal (thường là http://localhost:8501)
3. Tải lên file CSV của bạn bằng cách nhấn nút "Browse files"
4. Sử dụng giao diện PyGWalker để khám phá dữ liệu:
   - Kéo và thả các trường dữ liệu để tạo biểu đồ
   - Thay đổi loại biểu đồ
   - Lọc và tổng hợp dữ liệu

## Yêu cầu hệ thống

- Python 3.7+
- Trình duyệt web hiện đại (Chrome, Firefox, Edge...)

## Thư viện sử dụng

- Streamlit: Framework để xây dựng ứng dụng web data science
- PyGWalker: Công cụ khám phá dữ liệu tương tác
- Pandas: Thư viện phân tích dữ liệu
- Matplotlib & Plotly: Thư viện trực quan hóa dữ liệu 