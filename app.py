"""
Diabetes Early Detection System - Web Interface
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import seaborn as sns
from PIL import Image

# Page configuration settings
st.set_page_config(
    page_title="Diabetes Early Detection System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and related files
@st.cache_resource
def load_model():
    model = joblib.load('models/diabetes_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    features = joblib.load('models/features.pkl')
    return model, scaler, features

# Try to load the model
try:
    model, scaler, features = load_model()
    model_loaded = True
except FileNotFoundError:
    st.error("Model files not found. Please run the training script (diabetes.py) first.")
    model_loaded = False

# Interface title
st.title("Diabetes Early Detection System")
st.markdown("---")

# Create sidebar
st.sidebar.header("About this System")
st.sidebar.info(
    """
    This system uses machine learning to predict diabetes risk.

    **Features:**
    - Risk assessment based on clinical data
    - Personalized risk prediction
    - Visual risk assessment report
    - Batch prediction for multiple patients
    
    **Team:**   TiDuXiaFan
    """
)

# Add clinical integration guidelines to sidebar
st.sidebar.markdown("---")
st.sidebar.header("Clinical Integration Guidelines")
st.sidebar.markdown(
    """
    **For Healthcare Professionals:**
    1. Use as a supplementary diagnostic tool, not a replacement for clinical judgment
    2. High-risk predictions should trigger further clinical evaluation
    3. Consider patient history and family history for comprehensive assessment
    4. Re-evaluate moderate-risk patients every 6 months
    5. Update the system regularly to incorporate new clinical knowledge

    **Best Practice Workflow:**
    - Collect data during routine checkups
    - Use this system for preliminary risk screening
    - Schedule further tests for high-risk patients
    - Integrate results into electronic health records
    - Develop personalized prevention plans
    """
)

# Main content area with tabs
tab1, tab2, tab3 = st.tabs(["Risk Assessment", "Model Information", "Data Visualization"])

with tab1:
    st.header("Individual Risk Assessment")

    # Create two sub-tabs for individual and batch assessment
    individual_tab, batch_tab = st.tabs(["Individual Patient", "Batch Processing"])

    with individual_tab:
        st.markdown("Enter personal health metrics for diabetes risk evaluation.")

        # Create two-column layout
        col1, col2 = st.columns(2)

        # First column inputs
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=50, step=1)
            gender = st.selectbox("Gender", ["Male", "Female"])
            bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=24.0, format="%.1f", step=0.1)
            hba1c = st.number_input("HbA1c (%)", min_value=4.0, max_value=15.0, value=6.5, format="%.1f", step=0.1)
            chol = st.number_input("Total Cholesterol", min_value=2.0, max_value=10.0, value=4.2, format="%.1f",
                                   step=0.1)

        # Second column inputs
        with col2:
            urea = st.number_input("Urea", min_value=1.0, max_value=20.0, value=4.7, format="%.1f", step=0.1)
            cr = st.number_input("Creatinine (Cr)", min_value=20, max_value=200, value=46, step=1)
            tg = st.number_input("Triglycerides (TG)", min_value=0.2, max_value=10.0, value=0.9, format="%.1f",
                                 step=0.1)
            hdl = st.number_input("HDL Cholesterol", min_value=0.5, max_value=4.0, value=2.4, format="%.1f", step=0.1)
            ldl = st.number_input("LDL Cholesterol", min_value=0.5, max_value=5.0, value=1.4, format="%.1f", step=0.1)
            vldl = st.number_input("VLDL", min_value=0.1, max_value=3.0, value=0.5, format="%.1f", step=0.1)

        # Prediction button
        predict_btn = st.button("Evaluate Risk")

        # Display prediction results
        if predict_btn and model_loaded:
            # Prepare data
            gender_code = 1 if gender == "Male" else 0
            input_data = pd.DataFrame([[age, urea, cr, hba1c, chol, tg, hdl, ldl, vldl, bmi, gender_code]],
                                      columns=features)

            # Data standardization and prediction
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            prediction_prob = model.predict_proba(input_scaled)[0][1]

            # Display results
            st.markdown("---")
            st.subheader("Risk Assessment Results")

            # Display different risk levels and recommendations based on prediction probability
            if prediction_prob >= 0.7:
                risk_level = "High Risk"
                risk_color = "red"
                recommendations = """
                **Recommendations:**
                - Consult a doctor immediately for comprehensive diabetes testing
                - Monitor blood glucose levels
                - Consider dietary and lifestyle adjustments
                - Schedule regular follow-up tests
                """
            elif prediction_prob >= 0.4:
                risk_level = "Moderate Risk"
                risk_color = "orange"
                recommendations = """
                **Recommendations:**
                - Schedule diabetes screening within 3 months
                - Increase physical activity
                - Focus on healthy eating habits
                - Monitor health metrics regularly
                """
            else:
                risk_level = "Low Risk"
                risk_color = "green"
                recommendations = """
                **Recommendations:**
                - Maintain healthy lifestyle
                - Continue regular checkups
                - Focus on balanced diet
                - Maintain appropriate weight
                """

            # Display risk level and probability
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"<h1 style='text-align: center; color: {risk_color};'>{risk_level}</h1>",
                            unsafe_allow_html=True)
                st.markdown(f"<h3 style='text-align: center;'>Risk Probability: {prediction_prob:.1%}</h3>",
                            unsafe_allow_html=True)

            with col2:
                st.markdown("### Personalized Recommendations")
                st.markdown(recommendations)

                # Display key indicators to monitor
                st.markdown("### Key Indicators to Monitor")
                key_indicators = []

                if hba1c > 5.7:
                    key_indicators.append(f"- HbA1c: {hba1c}% - Above normal range (4.0-5.7%)")

                if bmi > 25:
                    key_indicators.append(f"- BMI: {bmi} - Above ideal range (18.5-24.9)")

                if tg > 1.7:
                    key_indicators.append(f"- Triglycerides: {tg} - Above normal range (0.4-1.7)")

                if not key_indicators:
                    st.markdown("All indicators are within normal ranges")
                else:
                    for indicator in key_indicators:
                        st.markdown(indicator)

    with batch_tab:
        st.markdown("Upload a CSV file with patient data for batch risk assessment.")

        st.markdown("""
        ### CSV Format Requirements
        Your CSV file should contain columns for all required health metrics:
        - AGE (years)
        - Gender (Male=1, Female=0, or use 'M'/'F')
        - BMI
        - HbA1c (%)
        - Chol (Total Cholesterol)
        - TG (Triglycerides)
        - HDL (HDL Cholesterol)
        - LDL (LDL Cholesterol)
        - VLDL
        - Urea
        - Cr (Creatinine)
        """)

        # Upload CSV file
        uploaded_file = st.file_uploader("Upload patient data CSV", type=['csv'])

        if uploaded_file is not None and model_loaded:
            # Load data
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"File uploaded successfully. {len(df)} records found.")

                # Display data preview
                st.subheader("Data Preview")
                st.dataframe(df.head())

                # Check if necessary columns exist
                missing_features = [f for f in features if f not in df.columns]

                if missing_features:
                    # Try to handle common column name differences
                    if 'Gender' in df.columns and 'gender' not in [col.lower() for col in df.columns]:
                        df['gender'] = df['Gender'].map({'M': 1, 'F': 0, 'Male': 1, 'Female': 0})
                        if 'Gender' in missing_features:
                            missing_features.remove('Gender')

                    # If features are still missing
                    if missing_features:
                        st.error(f"The uploaded CSV is missing required columns: {', '.join(missing_features)}")
                        st.info("Please ensure your CSV contains all the necessary health metrics.")
                else:
                    # Process data
                    process_btn = st.button("Process Batch")

                    if process_btn:
                        # Preprocess data
                        # Handle gender column (if in text format)
                        if 'Gender' in df.columns and df['Gender'].dtype == object:
                            df['Gender'] = df['Gender'].map({'M': 1, 'F': 0, 'Male': 1, 'Female': 0})

                        # Ensure all column names match model features
                        for col in features:
                            if col not in df.columns:
                                if col.lower() in [c.lower() for c in df.columns]:
                                    match = [c for c in df.columns if c.lower() == col.lower()][0]
                                    df[col] = df[match]

                        # Select needed features and convert data types
                        input_data = df[features].copy()
                        for col in input_data.columns:
                            if input_data[col].dtype == object:
                                input_data[col] = pd.to_numeric(input_data[col], errors='coerce')

                        # Standardize data
                        input_scaled = scaler.transform(input_data)

                        # Predictions
                        predictions = model.predict(input_scaled)
                        prediction_probs = model.predict_proba(input_scaled)[:, 1]

                        # Add prediction results to dataframe
                        results_df = df.copy()
                        results_df['Diabetes_Risk'] = predictions
                        results_df['Risk_Probability'] = prediction_probs

                        # Add risk level
                        def get_risk_level(prob):
                            if prob >= 0.7:
                                return "High Risk"
                            elif prob >= 0.4:
                                return "Moderate Risk"
                            else:
                                return "Low Risk"

                        results_df['Risk_Level'] = results_df['Risk_Probability'].apply(get_risk_level)

                        # Display results
                        st.subheader("Batch Processing Results")
                        st.dataframe(results_df)

                        # Risk distribution visualization
                        st.subheader("Risk Distribution")
                        fig, ax = plt.subplots(figsize=(10, 6))

                        # Ensure risk levels are sorted by severity
                        ordered_levels = ['Low Risk', 'Moderate Risk', 'High Risk']
                        risk_counts = results_df['Risk_Level'].value_counts().reindex(ordered_levels).fillna(0)

                        # Color mapping
                        colors = {'High Risk': 'red', 'Moderate Risk': 'orange', 'Low Risk': 'green'}
                        bar_colors = [colors[x] for x in risk_counts.index]

                        # Create bar chart
                        bars = ax.bar(risk_counts.index, risk_counts.values, color=bar_colors)

                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width() / 2., height + 5,
                                    f'{int(height)}',
                                    ha='center', va='bottom', fontsize=11, fontweight='bold')

                        ax.set_title('Distribution of Diabetes Risk Levels', fontsize=14)
                        ax.set_ylabel('Number of Patients', fontsize=12)
                        ax.set_ylim(0, max(risk_counts.values) * 1.15)
                        ax.grid(axis='y', linestyle='--', alpha=0.3)
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        st.pyplot(fig)

                        # Add text summary
                        st.markdown(f"""
                        **Risk Distribution Summary:**
                        - **High Risk**: {int(risk_counts.get('High Risk', 0))} patients ({(risk_counts.get('High Risk', 0) / risk_counts.sum() * 100):.1f}%)
                        - **Moderate Risk**: {int(risk_counts.get('Moderate Risk', 0))} patients ({(risk_counts.get('Moderate Risk', 0) / risk_counts.sum() * 100):.1f}%)
                        - **Low Risk**: {int(risk_counts.get('Low Risk', 0))} patients ({(risk_counts.get('Low Risk', 0) / risk_counts.sum() * 100):.1f}%)
                        """)

            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.info("Please check your CSV file format and try again.")

with tab2:
    st.header("Model Information")

    st.markdown("""
        <style>
        .card {
            border-radius: 5px;
            background-color: #f8f9fa;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .header {
            color: #2c3e50;
            margin-bottom: 15px;
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 10px;
        }
        </style>
        """, unsafe_allow_html=True)

    # Main model description card
    st.markdown("""
        <div class="card">
            <h3 class="header">About the Prediction Model</h3>
            <p>This system uses multiple machine learning algorithms to predict diabetes risk by analyzing various health indicators. We trained and compared three different models:</p>
            <ul>
                <li><strong>Logistic Regression</strong>: A baseline model for binary classification</li>
                <li><strong>Random Forest</strong>: An ensemble method that builds multiple decision trees</li>
                <li><strong>XGBoost</strong>: An advanced gradient boosting algorithm</li>
            </ul>
            <p>The models were evaluated using multiple metrics to ensure reliable performance, with the best performing model selected for deployment.</p>
        </div>
        """, unsafe_allow_html=True)

    # Split into two-column layout
    col1, col2 = st.columns(2)

    with col1:
        # Class distribution visualization
        st.markdown('<h3 class="header">Dataset Class Distribution</h3>', unsafe_allow_html=True)
        st.markdown("""
            This visualization shows the distribution of classes in our training dataset. 
            Our model is trained to distinguish between:
            - Non-diabetic patients (Class 0)
            - Pre-diabetic & Diabetic patients (Class 1)
            """)

        try:
            if os.path.exists('class_distribution.png'):
                img = Image.open('class_distribution.png')
                st.image(img, caption="Distribution of Diabetes Cases in Training Data", use_container_width=True)
        except Exception as e:
            st.error(f"Could not load class distribution image: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Confusion matrix visualization
        st.markdown('<h3 class="header">Model Performance: Confusion Matrix</h3>', unsafe_allow_html=True)
        st.markdown("""
            The confusion matrix shows how well our model classifies patients:
            - **True Positives**: Correctly identified diabetic patients
            - **True Negatives**: Correctly identified non-diabetic patients
            - **False Positives**: Non-diabetic patients incorrectly classified as diabetic
            - **False Negatives**: Diabetic patients incorrectly classified as non-diabetic
            """)

        try:
            if os.path.exists('confusion_matrix.png'):
                img = Image.open('confusion_matrix.png')
                st.image(img, caption="Confusion Matrix of Model Predictions", use_container_width=True)
        except Exception as e:
            st.error(f"Could not load confusion matrix image: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # Feature importance visualization
        st.markdown('<h3 class="header">Feature Importance Analysis</h3>', unsafe_allow_html=True)
        st.markdown("""
            This chart shows which health metrics are most important for predicting diabetes risk. 
            Higher values indicate greater influence on the model's predictions.

            Key influential factors include:
            - HbA1c (glycated hemoglobin): A measure of average blood sugar levels
            - BMI (body mass index): A measure of body fat based on height and weight
            - Other blood markers like cholesterol levels and kidney function markers
            """)

        try:
            if os.path.exists('feature_importance.png'):
                img = Image.open('feature_importance.png')
                st.image(img, caption="Feature Importance Analysis", use_container_width=True)
        except Exception as e:
            st.error(f"Could not load feature importance image: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

        # correlation heatmap visualization
        st.markdown('<h3 class="header">Feature Correlation Analysis</h3>', unsafe_allow_html=True)
        st.markdown("""
            The correlation heatmap reveals relationships between different health metrics:
            - Positive correlations (blue): Both metrics tend to increase together
            - Negative correlations (red): As one metric increases, the other tends to decrease
            - Strong correlations near ±1: Stronger relationships between metrics

            Understanding these relationships helps identify redundant information and potential risk factor combinations.
            """)

        try:
            if os.path.exists('correlation_heatmap.png'):
                img = Image.open('correlation_heatmap.png')
                st.image(img, caption="Feature Correlation Heatmap", use_container_width=True)
        except Exception as e:
            st.error(f"Could not load correlation heatmap image: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Display evaluation metrics explanation
    st.markdown("""
    ### Model Evaluation Metrics

    The model was evaluated using multiple metrics to ensure reliable performance:

    - **Accuracy**: Overall proportion of correct predictions
    - **Precision**: Proportion of true positives among predicted positives (minimizes false alarms)
    - **Recall**: Proportion of true positives correctly identified (minimizes missed cases)
    - **F1 Score**: Harmonic mean of precision and recall
    - **AUC-ROC**: Measures model's ability to distinguish between classes
    """)

# Data Visualization tab implementation
with tab3:
    st.header("Data Visualization")
    st.markdown("This section shows distributions and relationships of diabetes-related health indicators.")

    st.info("Please upload a CSV dataset to generate visualizations.")

    # File uploader for CSV files
    uploaded_file = st.file_uploader("Upload CSV file for visualization", type=['csv'])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            # Check if required columns exist in the uploaded CSV
            required_cols = ['HbA1c', 'CLASS']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                st.error(f"The uploaded CSV is missing required columns: {', '.join(missing_cols)}")
            else:
                if 'CLASS' in df.columns and df['CLASS'].dtype == object:
                    df['CLASS'] = df['CLASS'].map({'N': 0, 'P': 1, 'Y': 1})

                # Create HbA1c distribution visualization
                st.subheader("HbA1c Distribution")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(data=df, x='HbA1c', hue='CLASS', palette=['green', 'red'], kde=True, ax=ax)
                ax.set_title('HbA1c Distribution')
                ax.set_xlabel('HbA1c')
                ax.set_ylabel('Frequency')
                st.pyplot(fig)

                # Create correlation heatmap from the uploaded data
                st.subheader("Feature Correlation Analysis")
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                corr = df[numeric_cols].corr()
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
                ax.set_title('Feature Correlation Heatmap')
                st.pyplot(fig)

                # Add to the visualization part in tab3
                if 'CLASS' in df.columns:
                    st.subheader("Relationship Between Key Metrics and Diabetes Status")

                    # Let user select health metrics to view
                    available_metrics = [col for col in df.columns if col not in ['ID', 'No_Pation', 'CLASS']
                                         and df[col].dtype in [np.float64, np.int64]]
                    selected_metrics = st.multiselect(
                        "Select health metrics to compare",
                        options=available_metrics,
                        default=['HbA1c',
                                 'BMI'] if 'HbA1c' in available_metrics and 'BMI' in available_metrics else available_metrics[
                                                                                                            :2]
                    )

                    if selected_metrics:
                        # Create grouped boxplot
                        fig, ax = plt.subplots(figsize=(12, 6))
                        df_melt = pd.melt(df, id_vars=['CLASS'], value_vars=selected_metrics,
                                          var_name='Metric', value_name='Value')
                        df_melt['CLASS'] = df_melt['CLASS'].map({0: 'Non-diabetic', 1: 'Diabetic'})

                        sns.boxplot(x='Metric', y='Value', hue='CLASS', data=df_melt, palette=['green', 'red'], ax=ax)
                        ax.set_title('Distribution of Health Metrics by Diabetes Status', fontsize=16)
                        ax.set_xlabel('Health Metrics', fontsize=14)
                        ax.set_ylabel('Value', fontsize=14)
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)

        except Exception as e:
            st.error(f"Error processing file: {e}")

# Footer
st.markdown("---")