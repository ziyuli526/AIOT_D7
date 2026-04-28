import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import io
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Linear Regression CRISP-DM Demo",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Premium Look ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #00d4ff;
    }
    .crisp-header {
        font-weight: bold;
        color: #ffaa00;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar: Data Configuration ---
st.sidebar.header("🛠️ Data Configuration")

with st.sidebar:
    n_samples = st.slider("Number of samples (n)", 100, 1000, 500)
    variance = st.slider("Noise Variance (Var)", 0, 1000, 200)
    noise_mean = st.slider("Noise Mean", -10, 10, 0)
    seed = st.number_input("Random Seed", value=42, step=1)
    
    st.markdown("---")
    st.subheader("True Model Parameters")
    true_a = st.slider("True Slope (a)", -10.0, 10.0, 2.5)
    true_b = st.slider("True Intercept (b)", -50.0, 50.0, 10.0)
    
    generate_btn = st.button("🚀 Generate New Data", use_container_width=True)

# --- Data Generation Logic ---
@st.cache_data
def generate_data(n, a, b, n_mean, n_var, seed_val):
    np.random.seed(seed_val)
    x = np.random.uniform(-100, 100, n)
    noise = np.random.normal(n_mean, np.sqrt(n_var), n)
    y = a * x + b + noise
    df = pd.DataFrame({'Feature_X': x, 'Target_Y': y})
    return df, a, b

# Load Data
if 'data' not in st.session_state or generate_btn:
    df, actual_a, actual_b = generate_data(n_samples, true_a, true_b, noise_mean, variance, seed)
    st.session_state.data = df
    st.session_state.actual_params = (actual_a, actual_b)

df = st.session_state.data
actual_a, actual_b = st.session_state.actual_params

# --- App Header ---
st.title("📈 Linear Regression: CRISP-DM Workflow")
st.markdown("""
This application demonstrates a complete Machine Learning lifecycle for a Linear Regression problem, 
following the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology.
""")

# --- CRISP-DM Phases Tabs ---
tabs = st.tabs([
    "1. Business Understanding", 
    "2. Data Understanding", 
    "3. Data Preparation", 
    "4. Modeling", 
    "5. Evaluation", 
    "6. Deployment"
])

# --- Phase 1: Business Understanding ---
with tabs[0]:
    st.header("🏢 Phase 1: Business Understanding")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Objective")
        st.write("""
        The goal is to predict a continuous numerical value (**Target_Y**) based on a single input feature (**Feature_X**). 
        In a real-world scenario, this could represent predicting sales based on advertising spend or house prices based on square footage.
        """)
        st.info("🎯 **Goal:** Minimize prediction error and understand the relationship between variables.")
    with col2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/b/b9/CRISP-DM_Process_Diagram.png", caption="CRISP-DM Framework")

# --- Phase 2: Data Understanding ---
with tabs[1]:
    st.header("📊 Phase 2: Data Understanding")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Data Summary")
        st.write(df.describe())
        st.write(f"**Total Samples:** {len(df)}")
    
    with col2:
        st.subheader("Raw Data Visualization")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x='Feature_X', y='Target_Y', alpha=0.6, ax=ax, color='#00d4ff')
        ax.set_title("Feature X vs Target Y")
        st.pyplot(fig)

# --- Phase 3: Data Preparation ---
with tabs[2]:
    st.header("🧹 Phase 3: Data Preparation")
    st.write("Preparing data for the model: Splitting into Training/Testing sets and Scaling features.")
    
    test_size = st.slider("Test Set Size (%)", 10, 50, 20)
    
    X = df[['Feature_X']]
    y = df['Target_Y']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=seed)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    col1, col2 = st.columns(2)
    col1.success(f"Training Set: {len(X_train)} samples")
    col2.success(f"Test Set: {len(X_test)} samples")
    
    st.code("""
# Preparation Steps:
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
    """, language="python")

# --- Phase 4: Modeling ---
with tabs[3]:
    st.header("🤖 Phase 4: Modeling")
    
    if st.button("🚂 Train Linear Regression Model"):
        with st.spinner("Optimizing coefficients..."):
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            st.session_state.model = model
            st.session_state.scaler = scaler
            time.sleep(0.5)
            st.success("Model Training Complete!")
            
    if 'model' in st.session_state:
        model = st.session_state.model
        # Get learned parameters in original space
        # y = mX + c. Scaled X = (X - mean)/std. 
        # y = model.coef_ * (X - mean)/std + model.intercept_
        # y = (model.coef_/std) * X + (model.intercept_ - model.coef_*mean/std)
        std = np.sqrt(st.session_state.scaler.var_[0])
        mean = st.session_state.scaler.mean_[0]
        learned_a = model.coef_[0] / std
        learned_b = model.intercept_ - (model.coef_[0] * mean / std)
        
        st.subheader("Model Parameters")
        m1, m2 = st.columns(2)
        m1.metric("Learned Slope (a')", f"{learned_a:.4f}", f"{learned_a - actual_a:.4f} vs True")
        m2.metric("Learned Intercept (b')", f"{learned_b:.4f}", f"{learned_b - actual_b:.4f} vs True")

# --- Phase 5: Evaluation ---
with tabs[4]:
    st.header("🧪 Phase 5: Evaluation")
    
    if 'model' in st.session_state:
        y_pred = st.session_state.model.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("MSE", f"{mse:.2f}")
        col2.metric("RMSE", f"{rmse:.2f}")
        col3.metric("R² Score", f"{r2:.4f}")
        
        st.subheader("Regression Results")
        fig, ax = plt.subplots(figsize=(10, 5))
        plt.scatter(X_test, y_test, color='gray', alpha=0.5, label='Actual Data (Test)')
        
        # Sort X_test for a smooth regression line plot
        sort_idx = np.argsort(X_test.values.flatten())
        plt.plot(X_test.values[sort_idx], y_pred[sort_idx], color='red', linewidth=3, label='Regression Line')
        
        plt.xlabel("Feature X")
        plt.ylabel("Target Y")
        plt.legend()
        st.pyplot(fig)
    else:
        st.warning("Please train the model in Phase 4 first.")

# --- Phase 6: Deployment ---
with tabs[5]:
    st.header("🚀 Phase 6: Deployment")
    
    if 'model' in st.session_state:
        st.subheader("Real-time Prediction")
        user_input = st.number_input("Enter Feature X for Prediction:", value=0.0)
        
        scaled_input = st.session_state.scaler.transform([[user_input]])
        prediction = st.session_state.model.predict(scaled_input)[0]
        
        st.markdown(f"### Predicted Y: **{prediction:.4f}**")
        
        st.divider()
        st.subheader("Export Model")
        
        # Save to buffer
        buffer = io.BytesIO()
        model_data = {
            'model': st.session_state.model,
            'scaler': st.session_state.scaler,
            'params': {'a': learned_a, 'b': learned_b}
        }
        joblib.dump(model_data, buffer)
        
        st.download_button(
            label="💾 Download Model (.joblib)",
            data=buffer.getvalue(),
            file_name="linear_regression_model.joblib",
            mime="application/octet-stream"
        )
    else:
        st.warning("Please train the model to enable deployment features.")

# --- Footer ---
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and Scikit-Learn | Optimized for Speed & Clarity")
