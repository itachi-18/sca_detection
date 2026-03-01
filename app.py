import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. DATASET GENERATION (Simulation - same as before)
# ==========================================
@st.cache_resource  # Cache the data generation for performance
def generate_medical_data(n=500):
    np.random.seed(42)
    data = {
        "Hemoglobin_Level": np.random.normal(10, 2, n),
        "MCV": np.random.normal(85, 10, n),
        "MCH": np.random.normal(28, 5, n),
        "WBC_Count": np.random.normal(8000, 2000, n),
        "RBC_Count": np.random.normal(4.5, 1, n),
        "Sickle_Cell_Status": np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
    }
    df = pd.DataFrame(data)
    df.loc[df["Sickle_Cell_Status"] == 1, "Hemoglobin_Level"] -= 3
    return df


df = generate_medical_data()


# ==========================================
# 2. PREPROCESSING & MODEL TRAINING (Cached for efficiency)
# ==========================================
@st.cache_resource
def train_models(df):
    X = df.drop("Sickle_Cell_Status", axis=1)
    y = df["Sickle_Cell_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model A: Logistic Regression
    log_model = LogisticRegression(random_state=42)
    log_model.fit(X_train_scaled, y_train)

    # Model B: Support Vector Machine (SVM) - Often performs well in medical
    svm_model = SVC(probability=True, kernel="linear", random_state=42)
    svm_model.fit(X_train_scaled, y_train)

    return scaler, log_model, svm_model, X_test_scaled, y_test


scaler, log_model, svm_model, X_test_scaled, y_test = train_models(df)

# ==========================================
# STREAMLIT UI LAYOUT
# ==========================================

st.set_page_config(
    page_title="Sickle Cell Anemia Detection",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a better look
st.markdown(
    """
    <style>
    .main-header {
        font-size: 3.5em;
        font-weight: bold;
        color: #FF4B4B; /* Streamlit's primary red */
        text-align: center;
        margin-bottom: 20px;
        text-shadow: 2px 2px 5px rgba(0,0,0,0.2);
    }
    .subheader {
        font-size: 1.8em;
        color: #31333F;
        text-align: center;
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        font-size: 1.2em;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #e03c3c;
        transform: scale(1.02);
    }
    .stAlert {
        padding: 15px;
        border-radius: 8px;
        font-size: 1.1em;
    }
    .stAlert.success {
        background-color: #e6ffe6;
        border-left: 5px solid #4CAF50;
    }
    .stAlert.warning {
        background-color: #fff3e6;
        border-left: 5px solid #FF9800;
    }
    .stAlert.error {
        background-color: #ffe6e6;
        border-left: 5px solid #F44336;
    }
    .reportview-container .main .block-container{
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="main-header">🩸 Sickle Cell Anemia Detection System 🩸</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="subheader">An AI-powered tool for early and accurate diagnosis.</p>',
    unsafe_allow_html=True,
)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to", ["Dashboard", "Patient Prediction", "Model Evaluation", "About"]
)

if page == "Dashboard":
    st.header("📊 System Dashboard")
    st.write("Overview of the dataset and key insights.")

    st.subheader("Dataset Sample")
    st.dataframe(df.head())

    st.subheader("Sickle Cell Status Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x="Sickle_Cell_Status", data=df, palette="pastel", ax=ax)
    ax.set_xticklabels(["Healthy (0)", "Sickle Cell (1)"])
    ax.set_title("Distribution of Sickle Cell Status")
    st.pyplot(fig)

    st.subheader("Feature Correlation Heatmap")
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
    ax_corr.set_title("Medical Feature Correlation")
    st.pyplot(fig_corr)

    st.info(
        "The correlation heatmap helps us understand relationships between different blood parameters. Notice the strong negative correlation of 'Hemoglobin_Level' with 'Sickle_Cell_Status'."
    )

elif page == "Patient Prediction":
    st.header("🔬 New Patient Diagnosis")
    st.write(
        "Enter the patient's latest blood test parameters to get a predictive diagnosis."
    )

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Patient Vitals Input")
        hemoglobin = st.slider(
            "Hemoglobin Level (g/dL)",
            min_value=3.0,
            max_value=18.0,
            value=12.0,
            step=0.1,
        )
        mcv = st.slider(
            "MCV (fL)", min_value=60.0, max_value=120.0, value=85.0, step=0.1
        )
        mch = st.slider(
            "MCH (pg)", min_value=15.0, max_value=40.0, value=28.0, step=0.1
        )

    with col2:
        st.subheader("Additional Blood Parameters")
        wbc = st.slider(
            "WBC Count (cells/µL)",
            min_value=1000,
            max_value=20000,
            value=7500,
            step=100,
        )
        rbc = st.slider(
            "RBC Count (cells/µL)", min_value=2.0, max_value=6.0, value=4.5, step=0.01
        )

    patient_data = {
        "Hemoglobin_Level": hemoglobin,
        "MCV": mcv,
        "MCH": mch,
        "WBC_Count": wbc,
        "RBC_Count": rbc,
    }

    st.markdown("---")
    if st.button("Get Diagnosis"):
        # Create a DataFrame for the single patient input
        input_df = pd.DataFrame([patient_data])

        # Scale the input data using the *trained* scaler
        scaled_input = scaler.transform(input_df)

        # Get predictions from both models
        log_pred = log_model.predict(scaled_input)[0]
        log_proba = log_model.predict_proba(scaled_input)[0]

        svm_pred = svm_model.predict(scaled_input)[0]
        svm_proba = svm_model.predict_proba(scaled_input)[0]

        st.subheader("🩺 Diagnosis Result")

        # Display results from the better-performing model (assuming SVM for this example)
        if svm_pred == 1:
            st.error(f"**Sickle Cell Anemia Detected!**")
            st.warning(
                f"Based on our analysis, there is a **{svm_proba[1]*100:.2f}%** probability of Sickle Cell Anemia."
            )
            st.write("---")
            st.info(
                "Please consult with a medical professional for further tests and confirmation. Early detection is crucial."
            )
        else:
            st.success(f"**No Sickle Cell Anemia Detected.**")
            st.info(
                f"Based on our analysis, there is a **{svm_proba[0]*100:.2f}%** probability of a healthy status."
            )
            st.write("---")
            st.info(
                "While the system indicates a low probability, always consult a medical professional if symptoms persist or for routine check-ups."
            )

        st.write("---")
        st.subheader("Detailed Model Predictions")
        st.markdown(
            f"**Logistic Regression:** {'POSITIVE' if log_pred == 1 else 'NEGATIVE'} (Confidence: {log_proba[1]*100:.2f}%)"
        )
        st.markdown(
            f"**Support Vector Machine (SVM):** {'POSITIVE' if svm_pred == 1 else 'NEGATIVE'} (Confidence: {svm_proba[1]*100:.2f}%)"
        )


elif page == "Model Evaluation":
    st.header("📈 Model Performance Metrics")
    st.write("Detailed evaluation of the trained Machine Learning models.")

    # Get predictions on the test set
    log_test_pred = log_model.predict(X_test_scaled)
    svm_test_pred = svm_model.predict(X_test_scaled)

    st.subheader("Logistic Regression Evaluation")
    st.text(f"Accuracy: {accuracy_score(y_test, log_test_pred):.2f}")
    st.text("Classification Report:")
    st.code(classification_report(y_test, log_test_pred))
    fig_log_cm, ax_log_cm = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        confusion_matrix(y_test, log_test_pred),
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax_log_cm,
    )
    ax_log_cm.set_title("Logistic Regression Confusion Matrix")
    ax_log_cm.set_xlabel("Predicted")
    ax_log_cm.set_ylabel("True")
    st.pyplot(fig_log_cm)
    st.markdown("---")

    st.subheader("Support Vector Machine (SVM) Evaluation")
    st.text(f"Accuracy: {accuracy_score(y_test, svm_test_pred):.2f}")
    st.text("Classification Report:")
    st.code(classification_report(y_test, svm_test_pred))
    fig_svm_cm, ax_svm_cm = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        confusion_matrix(y_test, svm_test_pred),
        annot=True,
        fmt="d",
        cmap="Greens",
        ax=ax_svm_cm,
    )
    ax_svm_cm.set_title("SVM Confusion Matrix")
    ax_svm_cm.set_xlabel("Predicted")
    ax_svm_cm.set_ylabel("True")
    st.pyplot(fig_svm_cm)
    st.markdown("---")
    st.info(
        "The confusion matrix helps us understand False Positives and False Negatives, which are critical in medical diagnostics."
    )

elif page == "About":
    st.header("ℹ️ About This Project")
    st.write(
        "This project demonstrates an AI-powered system for the early detection of Sickle Cell Anemia using machine learning."
    )

    st.subheader("Project Objective")
    st.write(
        """
        The primary objective is to develop an intelligent system that can detect Sickle Cell Anemia at an early stage using routine blood test parameters. This aims to assist doctors and healthcare providers in making faster and more accurate diagnostic decisions, ultimately improving patient outcomes.
    """
    )

    st.subheader("Key Features")
    st.markdown(
        """
    - **Interactive Dashboard:** Visualize dataset statistics and distributions.
    - **Real-time Prediction:** Input patient blood parameters and receive an instant diagnosis with confidence scores.
    - **Model Comparison:** Evaluate and compare the performance of Logistic Regression and Support Vector Machine (SVM) models using standard metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).
    - **Scalable Architecture:** Designed with maintainability and potential future expansion in mind.
    - **User-Friendly Interface:** Built with Streamlit for an intuitive web experience.
    """
    )

    st.subheader("Technology Stack")
    st.markdown(
        """
    - **Programming Language:** Python
    - **Core Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
    - **Web Framework:** Streamlit
    """
    )

    st.subheader("Disclaimer")
    st.warning(
        """
    This system is for **demonstration and educational purposes only** and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for any health concerns.
    """
    )

    st.markdown("---")
    st.write(
        "Developed as a Master Project for academic submission and a portfolio showcase."
    )

st.sidebar.markdown("---")
st.sidebar.markdown("Project by: Your Name/Team Name")
st.sidebar.markdown("Version 1.0")
