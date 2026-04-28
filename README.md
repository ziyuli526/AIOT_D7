網站連結 : http://localhost:8501/

<img width="2526" height="1135" alt="image" src="https://github.com/user-attachments/assets/a1bba3ac-e2e0-4abb-8dd3-5c6de79c645f" />


# AIOT_D7: Linear Regression CRISP-DM Demo

This project is a Streamlit-based web application that demonstrates the complete **CRISP-DM** (Cross-Industry Standard Process for Data Mining) workflow for a Linear Regression model.

## 🚀 Features
- **Interactive Data Generation**: Generate synthetic data ($y = ax + b + noise$) with custom parameters via sidebar sliders.
- **CRISP-DM Workflow**: Explore the 6 phases of data mining:
  1. **Business Understanding**: Define goals and objectives.
  2. **Data Understanding**: Statistical analysis and raw data visualization.
  3. **Data Preparation**: Data splitting and feature scaling using `StandardScaler`.
  4. **Modeling**: Train a Scikit-learn `LinearRegression` model.
  5. **Evaluation**: Analyze performance metrics (MSE, RMSE, $R^2$) and regression plots.
  6. **Deployment**: Real-time prediction interface and model export (`.joblib`).

## 🛠️ Installation
Ensure you have Python installed, then install the dependencies:
```bash
pip install streamlit scikit-learn pandas numpy matplotlib seaborn joblib
```

## 🏃 How to Run
```bash
streamlit run app.py
```

## 📁 Project Structure
- `app.py`: Main Streamlit application.
- `dialogue.md`: Development log (Traditional Chinese).
- `README.md`: Project documentation.

---
Built as part of the AIoT Smart Internet of Things course.
