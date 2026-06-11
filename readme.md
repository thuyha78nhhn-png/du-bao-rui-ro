# 🛡️ Ứng dụng Phát hiện Giao dịch Gian lận & Rủi ro tín dụng (Streamlit ML App)

Ứng dụng web được chuyển đổi tự động từ Notebook huấn luyện và đánh giá mô hình học máy (`phat_hien_giao_dich_gian_lan.ipynb`). Hệ thống tích hợp giao diện trực quan hóa, cho phép thay đổi siêu tham số của nhiều thuật toán khác nhau, phân tích mô hình đồng thời hỗ trợ dự báo rủi ro giao dịch trực tiếp hoặc theo lô lớn dữ liệu.

## ✨ Các tính năng chính của ứng dụng
Ứng dụng được chia làm 4 phân vùng chức năng thông qua Tab chính:
1. **📊 Tổng quan dữ liệu:** Đọc, kiểm tra kích thước và tổng hợp thống kê mô tả nhanh các biến đặc trưng dữ liệu.
2. **📈 Trực quan hóa đặc trưng:** Xem phân phối tần suất của biến mục tiêu (`default`) và các đặc trưng đầu vào bằng biểu đồ Plotly tương tác thông minh.
3. **🎯 Kết quả & Đánh giá mô hình:** Hiển thị trực quan Ma trận nhầm lẫn (Confusion Matrix) và bảng Báo cáo phân loại chi tiết (`Accuracy`, `Precision`, `Recall`, `F1-Score`) dựa trên mô hình cấu hình tại thanh Sidebar.
4. **🔮 Hệ thống Dự báo rủi ro:** Chạy kiểm tra rủi ro qua 2 chế độ: Nhập trực tiếp các chỉ số đặc trưng hoặc Tải lên file chứa danh sách giao dịch mới để dự báo tự động và xuất báo cáo (CSV).

## 🗂️ Cấu trúc tệp dữ liệu đầu vào yêu cầu
- **Tệp dữ liệu huấn luyện:** Cần chứa 14 cột biến độc lập liên tục đặt tên lần lượt từ `X_1` đến `X_14` và biến phân loại nhị phân mục tiêu là `default` (Nhận giá trị `0` cho giao dịch bình thường và `1` cho giao dịch có dấu hiệu gian lận).
- **Tệp dữ liệu dự báo hàng loạt:** Định dạng tương tự, bắt buộc chứa đầy đủ 14 trường thông tin đặc trưng giao dịch từ `X_1` đến `X_14` (Cấu trúc tương đương như file `X_new.xlsx`).

## ⚙️ Hướng dẫn cài đặt và khởi chạy ứng dụng

### Bước 1: Khởi tạo và kích hoạt môi trường ảo (Khuyến nghị)
```bash
# Trên Windows
python -m venv venv
venv\Scripts\activate

# Trên macOS/Linux
python3 -m venv venv
source venv/bin/activate
