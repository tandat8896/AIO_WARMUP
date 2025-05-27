import numpy as np 
import streamlit as st 
import matplotlib.pyplot as plt 
import pandas as pd 
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.preprocessing import StandardScaler , LabelEncoder
from sklearn.model_selection import train_test_split
import io 



st.set_page_config(page_title="Data Analysis & ML Models", page_icon="📊", layout="wide", initial_sidebar_state="expanded")



st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        transform: translateY(-2px);
    }
    h1, h2, h3 {
        color: #ff4b4b;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 2rem;
        background-color: #f0f2f6;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)



class Linearregression:
    def __init__(self,learning_rate = 0.01):
        self.learning_rate = learning_rate
        self.w = None 
        self.b = None
        self.losses = []

    def initialize_parameter(self):
        self.w = 0
        self.b = 0 
    
    def forward(self,X):
        return X* self.w + self.b
    
    def compute_loss(self , y_true , y_pred):
        return np.mean((y_pred-y_true)**2)
    
    def train_step(self,X,y):
        y_pred = self.forward(X)
        m = len(X)
        dw = (2/m) * np.sum(X*(y_pred-y))
        db = (2/m) * np.sum(y_pred - y)

        self.w  -= self.learning_rate * dw
        self.b  -= self.learning_rate * db
        loss = self.compute_loss(y,y_pred)
        self.losses.append(loss)
        return loss
    

    
    def predict(self , X):
        return self.forward(X)

@st.cache_data
#annotation của streamlit tự định nghĩa 
def load_advertising_data():
    return pd.read_csv('advertising.csv')

with st.sidebar:
    st.title("Linear Regression With Đạt")
    st.markdown("---")
    dataset_choice = st.radio("Chose dataset",
                                ["Advertising(Linearregression)"])

st.title("Data analysis & Machine Learning")
st.markdown("---")
if "Advertising" in dataset_choice:
    data = load_advertising_data()
    tabs = st.tabs(["Data Overview", "EDA", "Linear Regression"])
    with tabs[0]:
        st.header("Data Overview")
        col1,col2,col3=st.columns(3)
        with col1:
            st.metric("Total Records", len(data))
        with col2:
            st.metric("Average TV Budget",f"${data['TV'].mean():,.2f}")
        with col3:
            st.metric("Average Sales", f"${data['Sales'].mean():,.2f}")
        
        st.dataframe(data.head(10),use_container_width=True)
        st.markdown("### Sumary Statistics")
        st.dataframe(data.describe(), use_container_width=True)
    with tabs[1]:
        st.header("Exploratory Data Analysis")
        col1,col2 = st.columns(2)
        with col1:
            fig_tv = px.histogram(data,x='TV',
                                        title ='TV Advertising Budget Distribution',
                                        template= "plotly_white")
            st.plotly_chart(fig_tv,use_container_width=True)

        with col2:
            fig_sales = px.histogram(data , x='Sales',
                                    title = 'Sales Distribution',
                                    template="plotly_white")
            st.plotly_chart(fig_sales , use_container_width=True)

        fig_scatter = px.scatter(data , x='TV',y='Sales',
                                    title = 'TV Advertising vs Sales',
                                    template = "plotly_white")
        st.plotly_chart(fig_scatter,use_container_width=True)
    with tabs[2]:
        st.header("Linear Regression Model")

        col1 , col2 , col3 = st.columns(3)
        with col1:
            test_size = st.slider("Test Size",0.1,0.5,0.2,0.05)
        with col2:
            learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.001, value=0.001, step=0.0001, format="%.4f")

        with col3:
            n_epochs = st.number_input("Number of Epochs", min_value=10, max_value=2000, value=1000)


        if st.button("TrainModel"):
            X = data['TV'].values
            y = data['Sales'].values
            X_train , X_test , y_train, y_test = train_test_split(X,y,test_size=test_size,random_state=42)


            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train.reshape(-1,1)).flatten()
            X_test_scaled = scaler.transform(X_test.reshape(-1, 1)).flatten()

            model = Linearregression(learning_rate=learning_rate)
            model.initialize_parameter()

            col1,col2 = st.columns(2)
            loss_plot = col1.empty()
            pred_loss = col2.empty()
            metrics = st.empty()

            for epoch in range(n_epochs):
                loss = model.train_step(X_train_scaled,y_train)
            
                if epoch % 10 == 0:
                    with col1:
                        # Enhanced loss plot
                        fig_loss = go.Figure()
                        fig_loss.add_trace(
                            go.Scatter(
                                y=model.losses,
                                mode='lines',
                                name='Loss',
                                line=dict(color='#ff4b4b', width=2),
                                fill='tozeroy',
                                fillcolor='rgba(255,75,75,0.1)'
                            )
                        )
                        fig_loss.update_layout(
                            title=dict(
                                text='Training Progress',
                                y=0.95,
                                x=0.5,
                                xanchor='center',
                                yanchor='top'
                            ),
                            xaxis_title='Epoch',
                            yaxis_title='Loss',
                            template='plotly_white',
                            hovermode='x unified',
                            showlegend=False
                        )
                        loss_plot.plotly_chart(fig_loss, use_container_width=True)

                    
                    with col2:
                        # Enhanced prediction plot
                        fig_pred = go.Figure()
                        
                        # Add training points
                        fig_pred.add_trace(
                            go.Scatter(
                                x=X_train,
                                y=y_train, 
                                mode='markers',
                                name='Training Data',
                                marker=dict(
                                    size=8,
                                    color='#2E86C1',
                                    symbol='circle',
                                    line=dict(color='white', width=1)
                                )
                            )
                        )
                        
                        # Add test points if available
                        fig_pred.add_trace(
                            go.Scatter(
                                x=X_test,
                                y=y_test,
                                mode='markers',
                                name='Test Data',
                                marker=dict(
                                    size=8,
                                    color='#28B463',
                                    symbol='diamond',
                                    line=dict(color='white', width=1)
                                )
                            )
                        )
                        
                        # Add regression line
                        X_line = np.linspace(X_train.min(), X_train.max(), 100)
                        X_line_scaled = scaler.transform(X_line.reshape(-1, 1)).flatten()
                        y_line = model.predict(X_line_scaled)
                        
                        fig_pred.add_trace(
                            go.Scatter(
                                x=X_line,
                                y=y_line,
                                mode='lines',
                                name='Regression Line',
                                line=dict(color='#ff4b4b', width=2)
                            )
                        )
                        
                        fig_pred.update_layout(
                            title={
                                'text': 'Model Predictions',
                                'y':0.95,
                                'x':0.5,
                                'xanchor': 'center',
                                'yanchor': 'top'
                            },
                            xaxis_title='TV Advertising Budget ($)',
                            yaxis_title='Sales ($)',
                            template="plotly_white",
                            legend=dict(
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=0.01,
                                bgcolor='rgba(255,255,255,0.8)'
                            ),
                            hovermode='closest'
                        )
                        pred_loss.plotly_chart(fig_pred, use_container_width=True)

                    
                    # Calculate metrics
                    y_pred_train = model.predict(X_train_scaled)
                    y_pred_test = model.predict(X_test_scaled)
                    train_mse = np.mean((y_pred_train - y_train) ** 2)
                    test_mse = np.mean((y_pred_test - y_test) ** 2)
                    
                    # Enhanced metrics display
                    metrics.markdown(f"""
                    <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px;'>
                        <h3 style='color: #ff4b4b; margin-bottom: 15px;'>📊 Training Metrics</h3>
                        <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px;'>
                            <div>
                                <p style='color: #666; margin-bottom: 5px;'>Epoch</p>
                                <h4 style='margin: 0;'>{epoch + 1}</h4>
                            </div>
                            <div>
                                <p style='color: #666; margin-bottom: 5px;'>Training MSE</p>
                                <h4 style='margin: 0;'>{train_mse:.4f}</h4>
                            </div>
                            <div>
                                <p style='color: #666; margin-bottom: 5px;'>Test MSE</p>
                                <h4 style='margin: 0;'>{test_mse:.4f}</h4>
                            </div>
                            <div>
                                <p style='color: #666; margin-bottom: 5px;'>Model Parameters</p>
                                <h4 style='margin: 0;'>w: {model.w:.4f}, b: {model.b:.4f}</h4>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)