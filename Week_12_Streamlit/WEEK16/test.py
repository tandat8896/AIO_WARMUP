import plotly.graph_objects as go

# Dữ liệu giả lập
X_train = [1, 2, 3, 4, 5]
y_train = [1, 4, 9, 16, 25]

# Tạo biểu đồ scatter
fig = go.Figure()

# Dữ liệu gốc
fig.add_trace(go.Scatter(x=X_train, y=y_train, mode='markers', name='Data'))

# Hiển thị biểu đồ
fig.show()
