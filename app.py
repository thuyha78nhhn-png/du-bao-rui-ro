import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# --------------------------------------------------------------------------------
# LỆNH STREAMLIT ĐẦU TIÊN
# --------------------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="Hệ Thống Phát Hiện Giao Dịch Gian Lận",
    page_icon="🛡️"
)

# --------------------------------------------------------------------------------
# HÀM NẠP DỮ LIỆU DÙNG CHUNG CÓ CACHE
# --------------------------------------------------------------------------------
@st.cache_data
def load_data(file_bytes, file_name):
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
        elif file_name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            return None
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc file dữ liệu: {e}")
        return None

# --------------------------------------------------------------------------------
# THÀNH PHẦN 1: SIDEBAR — VÙNG CẤU HÌNH
# --------------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    # Tải dữ liệu huấn luyện
    uploaded_file = st.file_uploader(
        "Tải lên tệp dữ liệu huấn luyện (.csv, .xlsx)", 
        type=["csv", "xlsx"],
        help="Chọn tệp dữ liệu chứa các đặc trưng từ X_1 đến X_14 và biến mục tiêu 'default'."
    )
    
    st.divider()
    
    # Lựa chọn mô hình (Notebook có 3 mô hình)
    model_choice = st.selectbox(
        "Chọn mô hình phân loại",
        options=["Random Forest", "Decision Tree", "Logistic Regression"],
        help="Chọn thuật toán học máy muốn huấn luyện và đánh giá."
    )
    
    st.subheader("Tham số mô hình AI")
    
    # Hiển thị tham số động theo mô hình được lựa chọn dựa trên cấu hình trong notebook
    params = {}
    if model_choice == "Random Forest":
        params['n_estimators'] = st.slider("Số cây (n_estimators)", min_value=10, max_value=300, value=100, step=10, help="Số lượng cây quyết định trong rừng.")
        params['max_depth'] = st.slider("Độ sâu tối đa (max_depth)", min_value=1, max_value=30, value=10, help="Độ sâu giới hạn của từng cây quyết định.")
        params['random_state'] = st.number_input("Random State", value=42, step=1, help="Giá trị hạt giống để tái hiện kết quả.")
    
    elif model_choice == "Decision Tree":
        params['criterion'] = st.selectbox("Tiêu chí phân tách (criterion)", options=["gini", "entropy", "log_loss"], index=0, help="Hàm đo lường chất lượng phân tách.")
        params['max_depth'] = st.slider("Độ sâu tối đa (max_depth)", min_value=1, max_value=30, value=5, help="Độ sâu giới hạn của cây.")
        params['random_state'] = st.number_input("Random State", value=42, step=1, help="Giá trị hạt giống để tái hiện kết quả.")
        
    elif model_choice == "Logistic Regression":
        params['C'] = st.slider("Hệ số nghịch đảo điều hòa (C)", min_value=0.01, max_value=10.0, value=1.0, step=0.1, help="Giá trị C càng nhỏ, mức độ điều hòa càng mạnh.")
        params['max_iter'] = st.number_input("Số vòng lặp tối đa (max_iter)", value=1000, step=100, help="Số lượng vòng lặp tối đa cho thuật toán tối ưu.")
        params['random_state'] = st.number_input("Random State", value=42, step=1)

    st.divider()
    
    # Nút bấm hành động duy nhất để kích hoạt huấn luyện
    train_clicked = st.button("🚀 Huấn luyện mô hình", type="primary", use_container_width=True, help="Bấm để bắt đầu phân tách dữ liệu, chuẩn hóa và huấn luyện thuật toán đã chọn.")

# --------------------------------------------------------------------------------
# THÀNH PHẦN 2: HEADER — VÙNG ĐỊNH HƯỚNG
# --------------------------------------------------------------------------------
st.title("🛡️ Ứng dụng Phát hiện Giao dịch Gian lận & Rủi ro tín dụng")
st.caption("Ứng dụng hỗ trợ phân tích rủi ro hồ sơ và phát hiện gian lận giao dịch tự động dựa trên các thuật toán Học máy tối ưu từ Notebook.")

if uploaded_file is None:
    st.info("💡 Vui lòng tải file dữ liệu mẫu (.csv hoặc .xlsx) ở thanh Sidebar bên trái để bắt đầu khám phá.")
    st.stop()

# Đọc dữ liệu khi đã upload thành công
df_raw = load_data(uploaded_file.getvalue(), uploaded_file.name)

if df_raw is None:
    st.error("Không thể xử lý dữ liệu. Vui lòng kiểm tra lại định dạng file.")
    st.stop()

st.caption(f"📁 Đang sử dụng tệp dữ liệu huấn luyện: **{uploaded_file.name}**")
st.divider()

# Xác định danh sách biến đặc trưng đầu vào X và biến đích y dựa trên cấu trúc dữ liệu thực tế
feature_cols = [col for col in df_raw.columns if col.startswith('X_')]
target_col = 'default'

# Kiểm tra tính hợp lệ của cấu trúc file dữ liệu đầu vào
if not feature_cols or target_col not in df_raw.columns:
    st.error(f"Cấu trúc dữ liệu không khớp yêu cầu! Cần có các cột đặc trưng 'X_1' đến 'X_n' và cột mục tiêu là '{target_col}'.")
    st.stop()

# --------------------------------------------------------------------------------
# KHỐI XỬ LÝ HUẤN LUYỆN (ĐÃ FIX: Chỉ đưa feature_cols vào Scaler)
# --------------------------------------------------------------------------------
if train_clicked:
    with st.spinner("🔄 Đang xử lý tiền dữ liệu và huấn luyện mô hình..."):
        # CHỈ trích xuất các cột X_1 đến X_14 để đưa vào mô hình, bỏ qua các cột dư thừa khác
        X = df_raw[feature_cols]
        y = df_raw[target_col]
        
        # Phân tách tập dữ liệu
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=params.get('random_state', 42), stratify=y)
        
        # Tiền xử lý dữ liệu (Scaler) - CHỈ học và chuẩn hóa trên 14 cột đặc trưng này
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Khởi tạo mô hình tương ứng theo cấu hình lựa chọn
        if model_choice == "Random Forest":
            model = RandomForestClassifier(
                n_estimators=params['n_estimators'], 
                max_depth=params['max_depth'], 
                random_state=params['random_state']
            )
        elif model_choice == "Decision Tree":
            model = DecisionTreeClassifier(
                criterion=params['criterion'], 
                max_depth=params['max_depth'], 
                random_state=params['random_state']
            )
        elif model_choice == "Logistic Regression":
            model = LogisticRegression(
                C=params['C'], 
                max_iter=params['max_iter'], 
                random_state=params['random_state']
            )
            
        # Huấn luyện mô hình
        model.fit(X_train_scaled, y_train)
        
        # Dự báo trên tập kiểm định
        y_pred = model.predict(X_test_scaled)
        
        # Tính toán các chỉ tiêu hiệu năng kiểm định
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "cm": confusion_matrix(y_test, y_pred),
            "report": classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Lưu kết quả vào session_state để tái sử dụng ở mọi Tab mà không cần huấn luyện lại
        st.session_state['trained_model'] = model
        st.session_state['scaler'] = scaler
        st.session_state['metrics'] = metrics
        st.session_state['model_name'] = model_choice
        st.session_state['feature_cols'] = feature_cols
        
    st.success(f"🎉 Huấn luyện thành công mô hình **{model_choice}**!")

# --------------------------------------------------------------------------------
# KHỞI TẠO CÁC TABS CHỨC NĂNG CHÍNH
# --------------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Tổng quan dữ liệu", 
    "📈 Trực quan hóa đặc trưng", 
    "🎯 Kết quả & Đánh giá mô hình", 
    "🔮 Hệ thống Dự báo rủi ro"
])

# --------------------------------------------------------------------------------
# THÀNH PHẦN 3: TAB "TỔNG QUAN DỮ LIỆU"
# --------------------------------------------------------------------------------
with tab1:
    st.subheader("📊 Phân tích Thống kê Dữ liệu Thô")
    
    # Hiển thị số liệu kích thước dữ liệu tổng quan
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric("Tổng số dòng dữ liệu (Rows)", f"{df_raw.shape[0]:,}")
    with m_col2:
        st.metric("Tổng số cột dữ liệu (Columns)", f"{df_raw.shape[1]}")
    with m_col3:
        # Ước lượng dung lượng tệp tải lên
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.metric("Dung lượng tệp dữ liệu", f"{file_size_mb:.2f} MB")
        
    st.markdown("#### 📋 Xem trước 5 hàng đầu tiên của tập dữ liệu")
    st.dataframe(df_raw.head(5), use_container_width=True)
    
    st.markdown("#### 📉 Bảng mô tả thống kê các biến mô hình (X & y)")
    # Chỉ mô tả thống kê các biến đưa vào mô hình quyết định cấu trúc
    selected_cols_to_desc = feature_cols + [target_col]
    st.dataframe(df_raw[selected_cols_to_desc].describe().T, use_container_width=True)

# --------------------------------------------------------------------------------
# THÀNH PHẦN 4: TAB "TRỰC QUAN HÓA DỮ LIỆU"
# --------------------------------------------------------------------------------
with tab2:
    st.subheader("📈 Phân tích Phân phối Biến đặc trưng & Mục tiêu")
    
    # 1. Vẽ phân phối biến mục tiêu ưu tiên trước tiên
    fig_target = px.bar(
        df_raw[target_col].value_counts().reset_index(),
        x=target_col,
        y="count",
        color=target_col,
        labels={"default": "Trạng thái (0: Bình thường, 1: Gian lận)", "count": "Số lượng hồ sơ"},
        title="Biểu đồ phân phối tỉ lệ biến mục tiêu (default)",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig_target, use_container_width=True)
    
    st.markdown("#### Trực quan các đặc trưng giao dịch quan trọng")
    # Sử dụng multiselect để người dùng lựa chọn biến muốn xem phân phối (Tránh quá tải nếu có > 4 biến)
    default_features = feature_cols[:4] if len(feature_cols) >= 4 else feature_cols
    selected_features = st.multiselect(
        "Chọn các đặc trưng X cần vẽ biểu đồ phân phối:", 
        options=feature_cols, 
        default=default_features
    )
    
    if selected_features:
        # Bố trí biểu đồ dạng lưới cân đối tự động dựa trên số lượng được chọn
        cols = st.columns(2)
        for idx, feat in enumerate(selected_features):
            col_to_use = cols[idx % 2]
            with col_to_use:
                fig_feat = px.histogram(
                    df_raw, 
                    x=feat, 
                    color=target_col,
                    marginal="box",
                    title=f"Phân phối tần suất của biến đặc trưng {feat}",
                    barmode="overlay",
                    height=350,
                    color_discrete_sequence=px.colors.qualitative.Dark2
                )
                st.plotly_chart(fig_feat, use_container_width=True)
    else:
        st.warning("Vui lòng chọn ít nhất một đặc trưng để hiển thị biểu đồ phân phối.")

# --------------------------------------------------------------------------------
# THÀNH PHẦN 5: TAB "KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH MÔ HÌNH"
# --------------------------------------------------------------------------------
with tab3:
    st.subheader("🎯 Đánh giá Hiệu năng Chỉ số Mô hình sau Huấn luyện")
    
    # Điều phối: Kiểm tra xem mô hình đã được bấm nút huấn luyện hay chưa
    if 'trained_model' not in st.session_state:
        st.info("ℹ️ Mô hình chưa được cấu hình huấn luyện. Vui lòng chọn tham số và bấm nút **[🚀 Huấn luyện mô hình]** ở thanh bên trái (Sidebar).")
    else:
        metrics = st.session_state['metrics']
        model_name = st.session_state['model_name']
        
        st.markdown(f"#### Thuật toán hiện tại đang đánh giá: **{model_name}**")
        
        # Trình bày chỉ tiêu vô hướng hiệu năng chính
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Độ chính xác (Accuracy)", f"{metrics['accuracy']:.4%}")
        c2.metric("Độ chuẩn xác (Precision)", f"{metrics['precision']:.4%}")
        c3.metric("Độ nhạy (Recall / TPR)", f"{metrics['recall']:.4%}")
        c4.metric("Điểm F1-Score", f"{metrics['f1']:.4%}")
        
        st.divider()
        
        grid_col1, grid_col2 = st.columns(2)
        
        with grid_col1:
            st.markdown("##### 🧱 Ma trận nhầm lẫn (Confusion Matrix)")
            cm_data = metrics['cm']
            x_labels = ['Dự báo Bình thường (0)', 'Dự báo Gian lận (1)']
            y_labels = ['Thực tế Bình thường (0)', 'Thực tế Gian lận (1)']
            
            fig_cm = ff.create_annotated_heatmap(
                z=cm_data, 
                x=x_labels, 
                y=y_labels, 
                colorscale='Blues',
                showscale=True
            )
            fig_cm.update_layout(height=400)
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with grid_col2:
            st.markdown("##### 📄 Báo cáo phân loại chi tiết (Classification Report)")
            report_df = pd.DataFrame(metrics['report']).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='Purples', subset=['precision', 'recall', 'f1-score']), use_container_width=True)

# --------------------------------------------------------------------------------
# THÀNH PHẦN 6: TAB "SỬ DỤNG MÔ HÌNH"
# --------------------------------------------------------------------------------
with tab4:
    st.subheader("🔮 Chẩn đoán & Dự báo rủi ro Gian lận Giao dịch trực tuyến")
    
    if 'trained_model' not in st.session_state:
        st.info("ℹ️ Vui lòng huấn luyện mô hình thành công ở Sidebar trước khi sử dụng tính năng dự báo rủi ro thực tế.")
    else:
        model = st.session_state['trained_model']
        scaler = st.session_state['scaler']
        feature_cols = st.session_state['feature_cols']
        
        predict_mode = st.radio(
            "Chọn phương thức nhập dữ liệu kiểm tra:",
            options=["Nhập chỉ số trực tiếp của một giao dịch", "Tải tệp danh sách nhiều giao dịch (Dự báo hàng loạt)"],
            horizontal=True
        )
        
        # ------------------------------------------------------------------------
        # CHẾ ĐỘ 1 — NHẬP TRỰC TIẾP QUA FORM INPUT
        # ------------------------------------------------------------------------
        if predict_mode == "Nhập chỉ số trực tiếp của một giao dịch":
            st.markdown("##### ✏️ Điền các thông số kỹ thuật của giao dịch cần kiểm tra:")
            
            # Tạo form nhập dữ liệu động theo danh sách biến đặc trưng
            with st.form("single_prediction_form"):
                input_data = {}
                
                # Tính toán khoảng giá trị mặc định dựa trên dữ liệu thô để tối ưu trải nghiệm nhập liệu
                form_cols = st.columns(3)
                for idx, feat in enumerate(feature_cols):
                    col_target = form_cols[idx % 3]
                    min_val = float(df_raw[feat].min())
                    max_val = float(df_raw[feat].max())
                    mean_val = float(df_raw[feat].mean())
                    
                    with col_target:
                        input_data[feat] = st.number_input(
                            f"Thông số {feat}",
                            min_value=min_val - 10.0,
                            max_value=max_val + 10.0,
                            value=mean_val,
                            format="%.6f",
                            help=f"Giá trị thực tế trong tập mẫu dao động từ {min_val:.2f} tới {max_val:.2f}"
                        )
                
                submit_predict = st.form_submit_button("🔍 Tiến hành phân tích rủi ro", type="primary")
                
            if submit_predict:
                # Chuyển đổi dữ liệu input sang định dạng DataFrame
                single_df = pd.DataFrame([input_data])
                
                # Áp dụng bộ chuẩn hóa scaler từ lúc huấn luyện (Đã đồng bộ 14 cột đặc trưng)
                single_scaled = scaler.transform(single_df[feature_cols])
                
                # Dự báo kết quả
                prediction = model.predict(single_scaled)[0]
                proba = model.predict_proba(single_scaled)[0] if hasattr(model, "predict_proba") else None
                
                st.markdown("---")
                st.markdown("#### 📊 Kết quả chẩn đoán từ hệ thống:")
                
                if prediction == 1:
                    st.error("🚨 **CẢNH BÁO: Giao dịch có dấu hiệu rủi ro cao / Gian lận tín dụng!**")
                else:
                    st.success("✅ **AN TOÀN: Giao dịch được đánh giá bình thường, độ tin cậy an toàn cao.**")
                
                if proba is not None:
                    p_col1, p_col2 = st.columns(2)
                    p_col1.metric("Xác suất An toàn (Class 0)", f"{proba[0]:.2%}")
                    p_col2.metric("Xác suất Gian lận (Class 1)", f"{proba[1]:.2%}")
                    st.progress(float(proba[1]))
                    
        # ------------------------------------------------------------------------
        # CHẾ ĐỘ 2 — TẢI FILE DỰ BÁO HÀNG LOẠT (ĐÃ FIX KHỚP SỐ CỘT)
        # ------------------------------------------------------------------------
        elif predict_mode == "Tải tệp danh sách nhiều giao dịch (Dự báo hàng loạt)":
            st.markdown("##### 📁 Tải lên tệp dữ liệu cần chấm điểm rủi ro hàng loạt")
            st.caption("Yêu cầu: Tệp cấu trúc phải chứa đầy đủ các cột thuộc tính đặc trưng từ `X_1` đến `X_14` (Ví dụ như file `X_new.xlsx`).")
            
            bulk_file = st.file_uploader("Chọn file dữ liệu mới cần dự báo (.csv, .xlsx)", type=["csv", "xlsx"], key="bulk_uploader")
            
            if bulk_file is not None:
                bulk_df = load_data(bulk_file.getvalue(), bulk_file.name)
                
                if bulk_df is not None:
                    # Kiểm tra xem file upload có đủ các cột X không
                    missing_cols = [c for c in feature_cols if c not in bulk_df.columns]
                    
                    if missing_cols:
                        st.error(f"❌ Tệp tải lên thiếu các cột đặc trưng bắt buộc sau: {missing_cols}. Vui lòng kiểm tra lại cấu trúc dữ liệu đầu vào.")
                    else:
                        st.success("✅ Cấu trúc hợp lệ! Hệ thống đang tiến hành xử lý chấm điểm...")
                        
                        # FIX QUAN TRỌNG: Chỉ lấy chính xác 14 cột feature_cols đưa qua Scaler biến đổi cấu trúc
                        bulk_X = bulk_df[feature_cols]
                        bulk_scaled = scaler.transform(bulk_X)
                        
                        # Dự đoán nhãn rủi ro và xác suất bằng mô hình đã lưu
                        bulk_preds = model.predict(bulk_scaled)
                        
                        # Tạo DataFrame kết quả trả ra cho người dùng
                        result_df = bulk_df.copy()
                        result_df['Du_Bao_Rui_Ro_Default'] = bulk_preds
                        
                        if hasattr(model, "predict_proba"):
                            bulk_probs = model.predict_proba(bulk_scaled)[:, 1]
                            result_df['Xac_Suat_Gian_Lan_Percent'] = np.round(bulk_probs * 100, 2)
                        
                        # Thống kê số lượng kết quả dự báo được
                        num_fraud = int(np.sum(bulk_preds == 1))
                        num_total = len(bulk_preds)
                        
                        res_c1, res_c2 = st.columns(2)
                        res_c1.metric("Tổng số giao dịch đã quét", f"{num_total:,}")
                        res_c2.metric("Số giao dịch phát hiện rủi ro (Gian lận)", f"{num_fraud:,}", delta=f"Tỉ lệ: {(num_fraud/num_total):.2%}", delta_color="inverse")
                        
                        st.markdown("##### 📋 Bảng kết quả dự báo tổng hợp (Xem trước 100 dòng đầu)")
                        st.dataframe(result_df.head(100), use_container_width=True)
                        
                        # Tạo nút xuất dữ liệu kết quả ra file CSV để tải về máy
                        csv_buffer = io.StringIO()
                        result_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                        csv_output = csv_buffer.getvalue()
                        
                        st.download_button(
                            label="📥 Tải xuống toàn bộ bảng kết quả dự báo (.CSV)",
                            data=csv_output,
                            file_name="Ket_Qua_Du_Bao_Gian_Lan.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
