import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

st.set_page_config(
    page_title="Đồ thị hàm số",
    page_icon="📊",
    layout="wide"
)

st.title("Ứng dụng vẽ đồ thị hàm số")

# Tạo combo box để chọn bài tập
bai_tap = st.sidebar.selectbox(
    "Chọn bài tập:",
    ["Bài 1: Vẽ đồ thị hàm số cơ bản", 
     "Bài 2: So sánh 2 hàm số trên cùng một biểu đồ", 
     "Bài 3: Vẽ đồ thị hàm bậc 2", 
     "Bài 4: Tương tác với Slider để khảo sát đồ thị", 
     "Bài 5: Vẽ Heatmap cho hàm z = x² + y²"]
)

# Bài 1: Vẽ đồ thị hàm số cơ bản
if bai_tap == "Bài 1: Vẽ đồ thị hàm số cơ bản":
    st.header("Bài 1: Vẽ đồ thị hàm số cơ bản")
    st.write("Chọn 1 trong các hàm số sin, cos, exp, log và vẽ biểu đồ trên đoạn [-10, 10].")
    
    ham_so = st.selectbox(
        "Chọn hàm số:",
        ["sin", "cos", "exp", "log"]
    )
    
    x = np.linspace(-10, 10, 1000)
    
    if ham_so == "sin":
        y = np.sin(x)
        ten_ham = "f(x) = sin(x)"
    elif ham_so == "cos":
        y = np.cos(x)
        ten_ham = "f(x) = cos(x)"
    elif ham_so == "exp":
        y = np.exp(x)
        ten_ham = "f(x) = exp(x)"
    elif ham_so == "log":
        # Đảm bảo x > 0 cho hàm logarithm
        x_log = np.linspace(0.01, 10, 1000)
        y = np.log(x_log)
        x = x_log
        ten_ham = "f(x) = log(x)"
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, 'b-', linewidth=2)
    ax.set_title(f"Đồ thị hàm số {ten_ham}")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True)
    
    # Vẽ trục tọa độ
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    st.pyplot(fig)

# Bài 2: So sánh 2 hàm số trên cùng một biểu đồ
elif bai_tap == "Bài 2: So sánh 2 hàm số trên cùng một biểu đồ":
    st.header("Bài 2: So sánh 2 hàm số trên cùng một biểu đồ")
    st.write("Chọn hai hàm số bất kỳ trong số: sin, cos, exp, log, và vẽ chúng trên cùng một biểu đồ.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        ham_so_1 = st.selectbox(
            "Chọn hàm số thứ nhất:",
            ["sin", "cos", "exp", "log"]
        )
    
    with col2:
        ham_so_2 = st.selectbox(
            "Chọn hàm số thứ hai:",
            ["cos", "sin", "exp", "log"]
        )
    
    x = np.linspace(-10, 10, 1000)
    x_log = np.linspace(0.01, 10, 1000)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Vẽ hàm số thứ nhất
    if ham_so_1 == "sin":
        y1 = np.sin(x)
        ten_ham_1 = "f(x) = sin(x)"
        ax.plot(x, y1, 'b-', linewidth=2, label=ten_ham_1)
    elif ham_so_1 == "cos":
        y1 = np.cos(x)
        ten_ham_1 = "f(x) = cos(x)"
        ax.plot(x, y1, 'b-', linewidth=2, label=ten_ham_1)
    elif ham_so_1 == "exp":
        y1 = np.exp(x)
        ten_ham_1 = "f(x) = exp(x)"
        ax.plot(x, y1, 'b-', linewidth=2, label=ten_ham_1)
    elif ham_so_1 == "log":
        y1 = np.log(x_log)
        ten_ham_1 = "f(x) = log(x)"
        ax.plot(x_log, y1, 'b-', linewidth=2, label=ten_ham_1)
    
    # Vẽ hàm số thứ hai
    if ham_so_2 == "sin":
        y2 = np.sin(x)
        ten_ham_2 = "g(x) = sin(x)"
        ax.plot(x, y2, 'r-', linewidth=2, label=ten_ham_2)
    elif ham_so_2 == "cos":
        y2 = np.cos(x)
        ten_ham_2 = "g(x) = cos(x)"
        ax.plot(x, y2, 'r-', linewidth=2, label=ten_ham_2)
    elif ham_so_2 == "exp":
        y2 = np.exp(x)
        ten_ham_2 = "g(x) = exp(x)"
        ax.plot(x, y2, 'r-', linewidth=2, label=ten_ham_2)
    elif ham_so_2 == "log":
        y2 = np.log(x_log)
        ten_ham_2 = "g(x) = log(x)"
        ax.plot(x_log, y2, 'r-', linewidth=2, label=ten_ham_2)
    
    ax.set_title(f"So sánh đồ thị hàm số {ten_ham_1} và {ten_ham_2}")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x), g(x)")
    ax.grid(True)
    ax.legend()
    
    # Vẽ trục tọa độ
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    st.pyplot(fig)

# Bài 3: Vẽ đồ thị hàm bậc 2
elif bai_tap == "Bài 3: Vẽ đồ thị hàm bậc 2":
    st.header("Bài 3: Vẽ đồ thị hàm bậc 2")
    st.write("Nhập hệ số a, b, c cho phương trình y = ax² + bx + c và vẽ đồ thị tương ứng.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        a = st.number_input("Nhập hệ số a:", value=1.0)
    with col2:
        b = st.number_input("Nhập hệ số b:", value=0.0)
    with col3:
        c = st.number_input("Nhập hệ số c:", value=0.0)
    
    x = np.linspace(-10, 10, 1000)
    y = a * x**2 + b * x + c
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, 'g-', linewidth=2)
    ax.set_title(f"Đồ thị hàm số f(x) = {a}x² + {b}x + {c}")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True)
    
    # Vẽ trục tọa độ
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Tính và hiển thị đỉnh/đáy của parabol
    if a != 0:
        x_dinh = -b / (2 * a)
        y_dinh = a * x_dinh**2 + b * x_dinh + c
        ax.plot(x_dinh, y_dinh, 'ro')
        st.write(f"Điểm cực trị: ({x_dinh:.2f}, {y_dinh:.2f})")
        
        # Loại đường cong
        if a > 0:
            st.write("Đường parabol hướng lên (có cực tiểu)")
        else:
            st.write("Đường parabol hướng xuống (có cực đại)")
    
    st.pyplot(fig)

# Bài 4: Tương tác với Slider để khảo sát đồ thị
elif bai_tap == "Bài 4: Tương tác với Slider để khảo sát đồ thị":
    st.header("Bài 4: Tương tác với Slider để khảo sát đồ thị")
    st.write("Dùng slider để điều chỉnh giá trị của a, b, c và cập nhật đồ thị theo thời gian thực.")
    
    a = st.slider("Hệ số a:", min_value=-5.0, max_value=5.0, value=1.0, step=0.1)
    b = st.slider("Hệ số b:", min_value=-10.0, max_value=10.0, value=0.0, step=0.5)
    c = st.slider("Hệ số c:", min_value=-10.0, max_value=10.0, value=0.0, step=0.5)
    
    x = np.linspace(-10, 10, 1000)
    y = a * x**2 + b * x + c
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, 'g-', linewidth=2)
    ax.set_title(f"Đồ thị hàm số f(x) = {a}x² + {b}x + {c}")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True)
    
    # Vẽ trục tọa độ
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Tính và hiển thị đỉnh/đáy của parabol
    if a != 0:
        x_dinh = -b / (2 * a)
        y_dinh = a * x_dinh**2 + b * x_dinh + c
        ax.plot(x_dinh, y_dinh, 'ro')
        st.write(f"Điểm cực trị: ({x_dinh:.2f}, {y_dinh:.2f})")
        
        # Tính điểm cắt trục x
        delta = b**2 - 4*a*c
        if delta > 0:
            x1 = (-b + np.sqrt(delta)) / (2*a)
            x2 = (-b - np.sqrt(delta)) / (2*a)
            st.write(f"Điểm cắt trục x: x₁ = {x1:.2f} và x₂ = {x2:.2f}")
        elif delta == 0:
            x0 = -b / (2*a)
            st.write(f"Tiếp xúc trục x tại x = {x0:.2f}")
        else:
            st.write("Không có điểm cắt trục x")
    
    st.pyplot(fig)

# Bài 5: Vẽ Heatmap cho hàm z = x² + y²
elif bai_tap == "Bài 5: Vẽ Heatmap cho hàm z = x² + y²":
    st.header("Bài 5: Vẽ Heatmap cho hàm z = x² + y²")
    st.write("Dùng seaborn.heatmap để vẽ biểu đồ nhiệt của hàm z = x² + y².")
    
    range_val = st.slider("Phạm vi giá trị x, y:", min_value=1, max_value=10, value=5)
    resolution = st.slider("Độ phân giải:", min_value=10, max_value=100, value=50)
    
    x = np.linspace(-range_val, range_val, resolution)
    y = np.linspace(-range_val, range_val, resolution)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2
    
    # Hiển thị surface plot 3D
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Biểu đồ 3D")
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=False)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Đồ thị hàm số z = x² + y²')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Biểu đồ nhiệt (Heatmap)")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(Z, cmap="viridis", ax=ax)
        ax.set_title('Heatmap của hàm z = x² + y²')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        st.pyplot(fig)
        
    # Hiển thị contour plot
    st.subheader("Đường đồng mức (Contour)")
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contour(X, Y, Z, levels=15, cmap='viridis')
    ax.clabel(contour, inline=True, fontsize=8)
    ax.set_title('Đường đồng mức của hàm z = x² + y²')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    st.pyplot(fig)