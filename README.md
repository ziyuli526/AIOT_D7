網站連結 : http://localhost:8501/

<img width="2526" height="1135" alt="image" src="https://github.com/user-attachments/assets/a1bba3ac-e2e0-4abb-8dd3-5c6de79c645f" />

<img width="2554" height="1151" alt="image" src="https://github.com/user-attachments/assets/18f6a926-e024-4e73-bed5-2c633fa56032" />

<img width="2530" height="1155" alt="image" src="https://github.com/user-attachments/assets/6ce32f80-4d67-4068-a6a6-ebebeae649a6" />

<img width="2559" height="1140" alt="image" src="https://github.com/user-attachments/assets/0a023025-014d-475c-bcad-8fea2f6e783a" />

<img width="2493" height="1092" alt="image" src="https://github.com/user-attachments/assets/586b9a8b-54c7-4f95-a15b-0ed667e260af" />

<img width="2543" height="1126" alt="image" src="https://github.com/user-attachments/assets/de3d3df1-58d9-42fe-a1eb-f828909f4837" />


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
