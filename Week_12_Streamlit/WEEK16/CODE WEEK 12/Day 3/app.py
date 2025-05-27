import streamlit as st
import openai
import pandas as pd
from datetime import datetime

# Cấu hình trang
st.set_page_config(
    page_title="ChatGPT Demo",
    page_icon="💬",
    layout="wide"
)

# CSS tùy chỉnh
st.markdown("""
<style>
    /* Thiết lập màu nền trang */
    .main .block-container {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 0.5rem;
    }
    
    /* Định dạng tin nhắn chat */
    .chat-message {
        padding: 1.5rem; 
        border-radius: 0.8rem; 
        margin-bottom: 1rem; 
        display: flex;
        flex-direction: column;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        transition: all 0.2s ease;
    }
    
    .chat-message:hover {
        box-shadow: 0 3px 8px rgba(0,0,0,0.1);
    }
    
    .chat-message.user {
        background-color: #f8f9fa;
        color: #333333;
        border-bottom-right-radius: 0.2rem;
    }
    
    .chat-message.assistant {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        color: #333333;
        border-bottom-left-radius: 0.2rem;
    }
    
    .chat-message .message-content {
        display: flex;
        margin-top: 0.5rem;
    }
    
    .avatar {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 1rem;
        font-size: 1.2rem;
    }
    
    .message {
        flex: 1;
        line-height: 1.5;
    }
    
    .thinking {
        color: #666666;
        font-style: italic;
        padding: 0.8rem;
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        display: inline-block;
    }
    
    /* Tùy chỉnh sidebar */
    .css-6qob1r, .css-1544g2n {
        background-color: #f8f9fa !important;
        border-right: 1px solid #e9ecef;
    }
    
    /* Tùy chỉnh header và tiêu đề */
    h1, h2, h3 {
        color: #1e1e1e;
    }
    
    /* Tùy chỉnh input */
    .st-emotion-cache-1kyxreq, .st-bq {
        border: 1px solid #e6e6e6 !important;
        background-color: white !important;
        border-radius: 0.5rem !important;
    }
    
    /* Tùy chỉnh button */
    .stButton button {
        background-color: #4e7ecf !important;
        color: white !important;
        border-radius: 0.5rem !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton button:hover {
        background-color: #3a67b2 !important;
        box-shadow: 0 3px 6px rgba(0,0,0,0.1) !important;
    }
    
    /* Tùy chỉnh scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #c9c9c9;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #a8a8a8;
    }
</style>
""", unsafe_allow_html=True)

# Hàm để hiển thị tin nhắn chat
def display_message(role, content):
    if role == "user":
        with st.container():
            st.markdown(f"""
            <div class="chat-message user">
                <div class="message-content">
                    <div class="avatar">👤</div>
                    <div class="message">{content}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        with st.container():
            st.markdown(f"""
            <div class="chat-message assistant">
                <div class="message-content">
                    <div class="avatar">🤖</div>
                    <div class="message">{content}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Tiêu đề
st.title("💬 ChatGPT Demo App")
st.markdown("Ứng dụng demo đơn giản tương tác với OpenAI GPT-4o và GPT-4o mini")

# Phần cấu hình
with st.sidebar:
    st.header("Cấu hình")
    
    # Nhập API Key
    api_key = st.text_input("OpenAI API Key", type="password")
    
    # Chọn mô hình
    model = st.selectbox(
        "Chọn Model",
        ["gpt-4o", "gpt-4o-mini"]
    )
    
    # Thông tin thêm
    st.markdown("---")
    st.markdown("### Thông tin")
    st.markdown("""
    - OpenAI API Key là bắt buộc để sử dụng ứng dụng
    - GPT-4o: Mô hình mạnh hơn, chậm hơn
    - GPT-4o-mini: Mô hình nhanh hơn, giới hạn hơn
    """)

# Khởi tạo chat history trong session state nếu chưa có
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử tin nhắn
for message in st.session_state.messages:
    display_message(message["role"], message["content"])

# Nhập tin nhắn mới
prompt = st.chat_input("Nhập tin nhắn...")

# Xử lý tin nhắn mới
if prompt:
    # Kiểm tra API key
    if not api_key:
        st.error("Vui lòng nhập OpenAI API Key trong phần cấu hình!")
        st.stop()
    
    # Thêm tin nhắn vào lịch sử
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Hiển thị tin nhắn người dùng
    display_message("user", prompt)
    
    # Hiển thị thông báo "Đang suy nghĩ..."
    with st.container():
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("""
        <div class="thinking">Đang suy nghĩ...</div>
        """, unsafe_allow_html=True)
    
    try:
        # Gọi API OpenAI
        openai.api_key = api_key
        
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": m["role"], "content": m["content"]} 
                for m in st.session_state.messages
            ],
            temperature=0.7,
        )
        
        # Lấy phản hồi
        assistant_response = response.choices[0].message.content
        
        # Xóa thông báo "Đang suy nghĩ..."
        thinking_placeholder.empty()
        
        # Thêm phản hồi vào lịch sử
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        
        # Hiển thị phản hồi
        display_message("assistant", assistant_response)
        
    except Exception as e:
        # Xóa thông báo "Đang suy nghĩ..."
        thinking_placeholder.empty()
        st.error(f"Lỗi: {str(e)}")

# Thêm nút xóa lịch sử chat
if st.sidebar.button("Xóa lịch sử chat"):
    st.session_state.messages = []
    st.experimental_rerun() 