import streamlit as st
import pandas as pd
import pygwalker as pyg
from io import StringIO
import matplotlib.pyplot as plt
import plotly.express as px

# Thiết lập tiêu đề và giới thiệu ứng dụng
st.set_page_config(page_title="CSV Data Explorer with PyGWalker", layout="wide")

st.title("✨ CSV Data Explorer with PyGWalker")
st.markdown("""
Upload your CSV file and start exploring your data in an interactive environment.
PyGWalker provides Tableau-like drag-and-drop interface for visual data exploration.
""")

# Upload file CSV
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Đọc dữ liệu từ file CSV
    try:
        df = pd.read_csv(uploaded_file)
        
        # Hiển thị thông tin cơ bản về dữ liệu
        st.subheader("📊 Data Preview")
        st.dataframe(df.head())
        
        # Hiển thị thông tin thống kê
        st.subheader("📈 Data Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Basic Information:")
            st.write(f"- Rows: {df.shape[0]}")
            st.write(f"- Columns: {df.shape[1]}")
            st.write("- Column Types:")
            for col, dtype in df.dtypes.items():
                st.write(f"  - {col}: {dtype}")
                
        with col2:
            st.write("Summary Statistics:")
            st.dataframe(df.describe())
        
        # PyGWalker
        st.subheader("🔍 Interactive Data Exploration with PyGWalker")
        st.markdown("""
        Drag and drop columns to create visualizations. Use the interface below to explore your data.
        """)
        
        # Tạo và hiển thị PyGWalker
        pyg_html = pyg.to_html(df)
        st.components.v1.html(pyg_html, height=1000, scrolling=True)
        
    except Exception as e:
        st.error(f"Error: {e}")
        st.error("Please upload a valid CSV file.")
else:
    # Hiển thị hướng dẫn và mẫu dữ liệu
    st.info("Please upload a CSV file to get started with data exploration.")
    st.markdown("""
    ### How to use:
    1. Click the 'Browse files' button above
    2. Select a CSV file from your computer
    3. Wait for the data to load
    4. Use the PyGWalker interface to explore your data:
       - Drag fields to create visualizations
       - Change chart types
       - Filter and aggregate data
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit and PyGWalker for data exploration") 