import streamlit as st
import requests
import json
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Vui lòng cấu hình GOOGLE_API_KEY trong file .env")
    st.stop()

try:
    genai.configure(api_key=GOOGLE_API_KEY)
    
    # Set up the model
    generation_config = {
        "temperature": 0.7,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }

    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
    ]

    # Use the newer Gemini model
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    
    #st.info("Đã kết nối thành công với Gemini API")
except Exception as e:
    st.error(f"Lỗi khi khởi tạo Gemini API: {str(e)}")
    st.stop()

# API base URL
API_BASE_URL = "http://localhost:8000/api/v1"

def check_api_connection():
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

# Function to get patient information
def get_patient_info(patient_id):
    try:
        if not check_api_connection():
            return {"error": "Không thể kết nối đến server. Vui lòng kiểm tra:\n1. Server đã được khởi động chưa\n2. Server có đang chạy ở port 8000 không\n3. Firewall có chặn kết nối không"}
        
        response = requests.get(f"{API_BASE_URL}/patients/{patient_id}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        if isinstance(e, requests.exceptions.ConnectionError):
            return {"error": "Không thể kết nối đến server. Vui lòng kiểm tra server đã được khởi động chưa."}
        elif isinstance(e, requests.exceptions.HTTPError):
            if response.status_code == 404:
                return {"error": f"Không tìm thấy bệnh nhân với ID: {patient_id}"}
            return {"error": f"Lỗi server: {response.status_code} - {response.text}"}
        return {"error": f"Lỗi khi lấy thông tin bệnh nhân: {str(e)}"}

# Function to get patient visits
def get_patient_visits(patient_id):
    try:
        if not check_api_connection():
            return {"error": "Không thể kết nối đến server. Vui lòng kiểm tra server đã được khởi động chưa."}
            
        response = requests.get(f"{API_BASE_URL}/visits/patient/{patient_id}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        if isinstance(e, requests.exceptions.ConnectionError):
            return {"error": "Không thể kết nối đến server. Vui lòng kiểm tra server đã được khởi động chưa."}
        elif isinstance(e, requests.exceptions.HTTPError):
            if response.status_code == 404:
                return {"error": f"Không tìm thấy lịch sử khám bệnh cho ID: {patient_id}"}
            return {"error": f"Lỗi server: {response.status_code} - {response.text}"}
        return {"error": f"Lỗi khi lấy lịch sử khám bệnh: {str(e)}"}

# Function to get medical history
def get_medical_history(patient_id):
    try:
        if not check_api_connection():
            return {"error": "Không thể kết nối đến server. Vui lòng kiểm tra server đã được khởi động chưa."}
            
        response = requests.get(f"{API_BASE_URL}/medical-history/{patient_id}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        if isinstance(e, requests.exceptions.ConnectionError):
            return {"error": "Không thể kết nối đến server. Vui lòng kiểm tra server đã được khởi động chưa."}
        elif isinstance(e, requests.exceptions.HTTPError):
            if response.status_code == 404:
                return {"error": f"Không tìm thấy lịch sử y tế cho ID: {patient_id}"}
            return {"error": f"Lỗi server: {response.status_code} - {response.text}"}
        return {"error": f"Lỗi khi lấy lịch sử y tế: {str(e)}"}

# Function to search patients by name
def search_patients_by_name(name):
    try:
        if not check_api_connection():
            return {"error": "Không thể kết nối đến server. Vui lòng kiểm tra server đã được khởi động chưa."}
            
        response = requests.get(f"{API_BASE_URL}/patients/search", params={"name": name})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        if isinstance(e, requests.exceptions.ConnectionError):
            return {"error": "Không thể kết nối đến server. Vui lòng kiểm tra server đã được khởi động chưa."}
        elif isinstance(e, requests.exceptions.HTTPError):
            if response.status_code == 404:
                return {"error": f"Không tìm thấy bệnh nhân với tên: {name}"}
            return {"error": f"Lỗi server: {response.status_code} - {response.text}"}
        return {"error": f"Lỗi khi tìm kiếm bệnh nhân: {str(e)}"}

# Function to format patient data
def format_patient_data(patient):
    if isinstance(patient, list):
        return "\n\n".join([format_patient_data(p) for p in patient])
    
    if "error" in patient:
        return f"{patient['error']}"
    
    formatted = f"""
### Thông tin bệnh nhân
- **ID**: {patient.get('patient_id', 'N/A')}
- **Mã y tế**: {patient.get('medical_id', 'N/A')}
- **Họ và tên**: {patient.get('first_name', '')} {patient.get('last_name', '')}
- **Ngày sinh**: {patient.get('date_of_birth', 'N/A')}
- **Giới tính**: {patient.get('gender', 'N/A')}
- **Số điện thoại**: {patient.get('phone_number', 'N/A')}
- **Email**: {patient.get('email', 'N/A')}
- **Địa chỉ**: {patient.get('address', 'N/A')}
"""
    return formatted

# Function to format visit data
def format_visit_data(visits):
    if not visits:
        return "Không tìm thấy lịch sử khám bệnh."
    
    if "error" in visits:
        return f" {visits['error']}"
    
    formatted = "### Lịch sử khám bệnh\n"
    for visit in visits:
        formatted += f"""
#### Lần khám ngày {visit.get('visit_date', 'N/A')}
- **Bác sĩ**: {visit.get('doctor_id', 'N/A')}
- **Triệu chứng**: {visit.get('symptoms', 'N/A')}
- **Chẩn đoán**: {visit.get('diagnosis', 'N/A')}
- **Ghi chú**: {visit.get('notes', 'N/A')}
"""
    return formatted

# Function to format medical history
def format_medical_history(history):
    if not history:
        return "Không tìm thấy lịch sử y tế."
    
    if "error" in history:
        return f" {history['error']}"
    
    formatted = format_patient_data(history)
    if 'visits' in history:
        formatted += "\n" + format_visit_data(history['visits'])
    return formatted

# Available functions for Gemini
available_functions = {
    "get_patient_info": get_patient_info,
    "get_patient_visits": get_patient_visits,
    "get_medical_history": get_medical_history,
    "search_patients_by_name": search_patients_by_name
}

# Function to format response based on function name
def format_function_response(function_name, response):
    if function_name == "get_patient_info":
        formatted = format_patient_data(response)
    elif function_name == "get_patient_visits":
        formatted = format_visit_data(response)
    elif function_name == "get_medical_history":
        formatted = format_medical_history(response)
    elif function_name == "search_patients_by_name":
        formatted = format_patient_data(response)
    else:
        formatted = json.dumps(response, indent=2)
    
    # Thêm prompt vào đầu câu trả lời
    return f"Đây là thông tin mà bạn cần tìm  :\n\n{formatted}"

# Function to analyze user intent using Gemini
def analyze_intent(user_input):
    try:
        prompt = f"""
        Bạn là một trợ lý AI giúp phân tích yêu cầu của người dùng và chuyển đổi thành các lệnh gọi hàm.
        Hãy phân tích yêu cầu sau và trả về JSON với cấu trúc:
        {{
            "function_name": "tên_hàm",
            "parameters": {{
                "param1": "giá_trị1",
                "param2": "giá_trị2"
            }},
            "is_general_query": true/false
        }}

        Các hàm có sẵn:
        1. get_patient_info(patient_id) - Lấy thông tin bệnh nhân theo ID
        2. get_patient_visits(patient_id) - Lấy lịch sử khám bệnh
        3. get_medical_history(patient_id) - Lấy toàn bộ lịch sử y tế
        4. search_patients_by_name(name) - Tìm kiếm bệnh nhân theo tên

        Nếu câu hỏi không liên quan đến các chức năng trên, hãy đặt is_general_query là true.
        Nếu câu hỏi liên quan đến các chức năng trên, hãy đặt is_general_query là false và điền function_name và parameters phù hợp.

        Yêu cầu của người dùng: {user_input}

        Chỉ trả về JSON, không thêm text nào khác.
        """
        
        response = model.generate_content(prompt)
        
        if not response or not response.text:
            st.error("Không nhận được phản hồi từ Gemini API")
            return None
            
        # Clean the response text to ensure it's valid JSON
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        try:
            result = json.loads(response_text)
            if not isinstance(result, dict):
                st.error("Phản hồi không đúng định dạng yêu cầu")
                return None
            return result
        except json.JSONDecodeError as e:
            st.error(f"Không thể phân tích JSON: {str(e)}")
            st.error(f"Phản hồi nhận được: {response_text}")
            return None
            
    except Exception as e:
        st.error(f"Lỗi khi gọi Gemini API: {str(e)}")
        return None

# Function to get general AI response
def get_general_ai_response(user_input):
    try:
        prompt = f"""
        Bạn là một trợ lý AI thông minh và hữu ích. Hãy trả lời câu hỏi sau một cách tự nhiên và hữu ích:
        {user_input}
        """
        response = model.generate_content(prompt)
        return response.text if response and response.text else "Xin lỗi, tôi không thể tạo câu trả lời lúc này."
    except Exception as e:
        return f"Xin lỗi, đã xảy ra lỗi khi tạo câu trả lời: {str(e)}"

# Streamlit UI
st.title("Hệ thống EMR - Giao diện trò chuyện")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Mời nhập vào thông tin mà bạn cần tìm kiếm ?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Analyze intent using Gemini
    intent = analyze_intent(prompt)
    
    if intent:
        if intent.get("is_general_query", False):
            # Handle general queries with AI response
            ai_response = get_general_ai_response(prompt)
            st.session_state.messages.append({
                "role": "assistant",
                "content": ai_response
            })
            with st.chat_message("assistant"):
                st.markdown(ai_response)
        elif "function_name" in intent and "parameters" in intent:
            function_name = intent["function_name"]
            function_args = intent["parameters"]
            
            if function_name in available_functions:
                # Call the function
                function_response = available_functions[function_name](**function_args)
                
                # Format the response
                formatted_response = format_function_response(function_name, function_response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": formatted_response
                })
                
                with st.chat_message("assistant"):
                    st.markdown(formatted_response)
            else:
                error_message = "Thưa điện hạ Tấn Đạt, thần thiếp không thể hiểu được yêu cầu của Ngài. Vui lòng thử lại với cách diễn đạt khác."
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })
                with st.chat_message("assistant"):
                    st.markdown(error_message)
    else:
        error_message = "Thưa điện hạ Tấn Đạt, thần thiếp không thể hiểu được yêu cầu của Ngài. Vui lòng thử lại với cách diễn đạt khác."
        st.session_state.messages.append({
            "role": "assistant",
            "content": error_message
        })
        with st.chat_message("assistant"):
            st.markdown(error_message)
