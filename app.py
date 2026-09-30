import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_absolute_error
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor

# Page Configuration
st.set_page_config(
    page_title="EasyML Studio - 3-in-1 Predictor",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for clean, modern, and layman-friendly aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Header hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
        padding: 24px 28px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(49, 46, 129, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .hero-title {
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .hero-desc {
        font-size: 0.95rem;
        color: #C7D2FE;
        margin: 0;
        line-height: 1.5;
    }
    
    /* Step Cards */
    .step-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    
    .step-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .step-desc {
        font-size: 0.88rem;
        color: #94A3B8;
        margin-bottom: 12px;
    }
    
    /* Result Cards */
    .result-box-success {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.25) 100%);
        border: 1.5px solid #10B981;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        margin-top: 16px;
    }
    
    .result-box-warning {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.25) 100%);
        border: 1.5px solid #EF4444;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        margin-top: 16px;
    }
    
    .result-box-price {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(79, 70, 229, 0.25) 100%);
        border: 1.5px solid #6366F1;
        border-radius: 14px;
        padding: 22px;
        text-align: center;
        margin-top: 16px;
    }
    
    .result-title {
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
        color: #E2E8F0;
        margin-bottom: 6px;
    }
    
    .result-value {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 6px;
    }
    
    .result-details {
        font-size: 0.92rem;
        color: #CBD5E1;
    }
    
    /* Info helper callout */
    .helper-tip {
        background: rgba(59, 130, 246, 0.1);
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        font-size: 0.88rem;
        color: #93C5FD;
        margin-bottom: 14px;
    }
    
    /* Friendly metric pill */
    .metric-pill {
        display: inline-block;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 8px 14px;
        margin-right: 8px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# ----------------- CACHED DATA LOADERS ----------------- #

@st.cache_data
def get_diabetes_data():
    path = os.path.join(DATASET_DIR, "diabetes.csv")
    return pd.read_csv(path)

@st.cache_data
def get_cancer_data():
    path = os.path.join(DATASET_DIR, "breast_cancer_data.csv")
    df = pd.read_csv(path)
    if 'Unnamed: 32' in df.columns:
        df = df.drop(columns=['Unnamed: 32'])
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    if 'diagnosis' in df.columns:
        df['target'] = df['diagnosis'].map({'M': 1, 'B': 0})
        df = df.drop(columns=['diagnosis'])
    return df

@st.cache_data
def get_car_data():
    path = os.path.join(DATASET_DIR, "quikr_car.csv")
    car = pd.read_csv(path)
    car = car[car['year'].str.isnumeric()]
    car['year'] = car['year'].astype(int)
    car = car[car['Price'] != 'Ask For Price']
    car['Price'] = car['Price'].str.replace(',', '').astype(int)
    car['kms_driven'] = car['kms_driven'].str.split().str.get(0).str.replace(',', '')
    car = car[car['kms_driven'].str.isnumeric()]
    car['kms_driven'] = car['kms_driven'].astype(int)
    car = car[~car['fuel_type'].isna()]
    car['name'] = car['name'].str.split().str.slice(start=0, stop=3).str.join(' ')
    car = car[car['Price'] < 6e6].reset_index(drop=True)
    return car

# Auto-train default models in the background if not present
def ensure_default_models():
    # 1. Diabetes default SVM
    if 'diabetes' not in st.session_state.trained_models:
        df = get_diabetes_data()
        feat = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[feat])
        X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, df['Outcome'], test_size=0.2, stratify=df['Outcome'], random_state=2)
        clf = SVC(kernel='linear', probability=True, random_state=2)
        clf.fit(X_train, Y_train)
        st.session_state.trained_models['diabetes'] = {
            'model': clf,
            'scaler': scaler,
            'features': feat,
            'test_acc': accuracy_score(Y_test, clf.predict(X_test)),
            'algo': "Support Vector Machine (Recommended)"
        }

    # 2. Breast Cancer default Random Forest
    if 'cancer' not in st.session_state.trained_models:
        df = get_cancer_data()
        feat = [c for c in df.columns if c != 'target']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[feat])
        X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, df['target'], test_size=0.2, stratify=df['target'], random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, Y_train)
        st.session_state.trained_models['cancer'] = {
            'model': clf,
            'scaler': scaler,
            'features': feat,
            'test_acc': accuracy_score(Y_test, clf.predict(X_test)),
            'algo': "Random Forest Classifier (Recommended)"
        }

    # 3. Car Valuation default Linear Regression
    if 'car' not in st.session_state.trained_models:
        df = get_car_data()
        X = df[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
        y = df['Price']
        ohe = OneHotEncoder()
        ohe.fit(X[['name', 'company', 'fuel_type']])
        col_trans = make_column_transformer(
            (OneHotEncoder(categories=ohe.categories_), ['name', 'company', 'fuel_type']),
            remainder='passthrough'
        )
        pipe = make_pipeline(col_trans, LinearRegression())
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        st.session_state.trained_models['car'] = {
            'pipeline': pipe,
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'algo': "Standard Price Regression (Recommended)"
        }

ensure_default_models()

# ----------------- SIDEBAR ----------------- #

with st.sidebar:
    st.markdown("### 🌟 Choose an AI Tool")
    st.caption("Pick what you'd like to check or predict today:")
    
    app_choice = st.radio(
        "Navigation",
        [
            "🩺 Diabetes Risk Check",
            "🔬 Breast Tumor Diagnostic",
            "🚗 Used Car Price Estimator"
        ],
        label_visibility="collapsed"
    )
    
    st.divider()
    st.markdown("#### 💡 How it works")
    st.markdown("""
    1. **Fill in simple details** in the Predict tab.
    2. Click the **big action button** to get instant answers.
    3. Want to see under the hood? Check **Model & Data** tab anytime!
    """)
    st.divider()
    st.caption("⚡ Powered by Machine Learning & Python")


# ==============================================================================
# 1. DIABETES CHECK
# ==============================================================================
if app_choice == "🩺 Diabetes Risk Check":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🩺 Diabetes Health Risk Checker</div>
        <p class="hero-desc">Enter basic health parameters to instantly evaluate diabetes risk using trained machine learning.</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab_pred, tab_train, tab_data = st.tabs(["🔮 Check Your Risk (Simple)", "⚙️ Model & Training", "📊 Explore Dataset"])
    df_diabetes = get_diabetes_data()
    diab_feat = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    with tab_pred:
        st.markdown('<div class="step-header">Step 1: Enter Health Details</div>', unsafe_allow_html=True)
        st.markdown('<div class="step-desc">Fill in the values below. Hover over the ℹ️ icons if you are unsure of normal ranges.</div>', unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            glucose = st.slider("🍬 Blood Glucose Level (mg/dL)", min_value=50, max_value=250, value=115, 
                                help="Normal fasting glucose is typically under 100 mg/dL. 100-125 is prediabetes, 126+ indicates diabetes risk.")
            bmi = st.slider("⚖️ Body Mass Index (BMI in kg/m²)", min_value=12.0, max_value=60.0, value=26.5, step=0.1,
                            help="Healthy weight range is generally 18.5 - 24.9. 25-29.9 is overweight, 30+ is obese.")
            age = st.slider("🎂 Age (Years)", min_value=15, max_value=100, value=35)
            preg = st.number_input("🤰 Number of Pregnancies", min_value=0, max_value=20, value=1, step=1,
                                   help="Enter 0 for males or no previous pregnancies.")
            
        with c2:
            bp = st.slider("💓 Resting Blood Pressure (mm Hg)", min_value=40, max_value=150, value=75,
                           help="Normal diastolic pressure is usually between 60 - 80 mm Hg.")
            insulin = st.slider("💉 2-Hour Serum Insulin (mu U/ml)", min_value=0, max_value=400, value=80,
                                help="Normal 2-hour post-meal insulin is typically 16 - 166 mu U/ml.")
            skin = st.slider("📏 Skin Fold Thickness (mm)", min_value=0, max_value=80, value=23,
                             help="Triceps skin fold thickness measure (typically 10-30 mm).")
            dpf = st.slider("🧬 Family Diabetes History Score", min_value=0.05, max_value=2.50, value=0.45, step=0.01,
                            help="Higher value means stronger family history of diabetes.")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔍 Check Diabetes Risk Now", type="primary", use_container_width=True):
            user_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
            saved = st.session_state.trained_models['diabetes']
            user_scaled = saved['scaler'].transform(user_data)
            pred = saved['model'].predict(user_scaled)[0]
            
            prob = None
            if hasattr(saved['model'], "predict_proba"):
                prob = saved['model'].predict_proba(user_scaled)[0]
            
            if pred == 1:
                conf = f"Estimated Risk: **{prob[1]*100:.1f}% Likelihood**" if prob is not None else ""
                st.markdown(f"""
                <div class="result-box-warning">
                    <div class="result-title">⚠️ Assessment Result</div>
                    <div class="result-value" style="color: #F87171;">Elevated Diabetes Risk Detected</div>
                    <div class="result-details">
                        The model indicates high indicators for diabetes. {conf}<br>
                        <i>Please consult a certified healthcare professional or doctor for formal medical testing.</i>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                conf = f"Confidence: **{prob[0]*100:.1f}% Normal**" if prob is not None else ""
                st.markdown(f"""
                <div class="result-box-success">
                    <div class="result-title">✅ Assessment Result</div>
                    <div class="result-value" style="color: #34D399;">Low Risk / Likely Non-Diabetic</div>
                    <div class="result-details">
                        Health parameters are within normal indicators. {conf}<br>
                        <i>Maintain regular check-ups and a healthy lifestyle.</i>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with tab_train:
        st.markdown('<div class="step-header">Model Accuracy & Retraining</div>', unsafe_allow_html=True)
        saved = st.session_state.trained_models.get('diabetes', {})
        
        m1, m2 = st.columns(2)
        m1.info(f"**Current Model:** {saved.get('algo', 'SVM')}")
        m2.success(f"**Test Accuracy Score:** {saved.get('test_acc', 0.77)*100:.1f}%")
        
        st.markdown("#### Want to try a different algorithm?")
        algo_choice = st.selectbox(
            "Select ML Algorithm",
            [
                "Support Vector Machine (SVM Linear)",
                "Random Forest Classifier",
                "Logistic Regression",
                "Gradient Boosting"
            ],
            key="diab_user_algo"
        )
        
        if st.button("🚀 Retrain Diabetes Model", key="btn_retrain_diab"):
            with st.spinner("Retraining model..."):
                X = df_diabetes[diab_feat]
                Y = df_diabetes['Outcome']
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=2)
                
                if "SVM" in algo_choice:
                    clf = SVC(kernel='linear', probability=True, random_state=2)
                elif "Random Forest" in algo_choice:
                    clf = RandomForestClassifier(n_estimators=100, random_state=2)
                elif "Logistic" in algo_choice:
                    clf = LogisticRegression(random_state=2)
                else:
                    clf = GradientBoostingClassifier(random_state=2)
                
                clf.fit(X_train, Y_train)
                acc = accuracy_score(Y_test, clf.predict(X_test))
                st.session_state.trained_models['diabetes'] = {
                    'model': clf,
                    'scaler': scaler,
                    'features': diab_feat,
                    'test_acc': acc,
                    'algo': algo_choice
                }
                st.success(f"Retrained successfully! New Accuracy: {acc*100:.2f}%")

    with tab_data:
        st.markdown('<div class="step-header">About the Data</div>', unsafe_allow_html=True)
        st.markdown("This model is trained on the standard **PIMA Indians Diabetes Database** (768 patient records).")
        st.dataframe(df_diabetes.head(10), use_container_width=True)
        
        fig = px.histogram(df_diabetes, x="Glucose", color="Outcome", barmode="overlay",
                           title="Glucose Levels in Non-Diabetic (0) vs Diabetic (1) Patients",
                           color_discrete_map={0: "#10B981", 1: "#EF4444"},
                           labels={"Outcome": "Diabetes Status (0: No, 1: Yes)"},
                           template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# 2. BREAST CANCER DIAGNOSTIC
# ==============================================================================
elif app_choice == "🔬 Breast Tumor Diagnostic":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🔬 Breast Tumor Diagnostic AI</div>
        <p class="hero-desc">Evaluate whether a detected breast mass is Benign (non-cancerous) or Malignant (cancerous) based on cell measurements.</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab_pred, tab_train, tab_data = st.tabs(["🔮 Perform Diagnosis (Simple)", "⚙️ Model & Training", "📊 Explore Dataset"])
    df_cancer = get_cancer_data()
    cancer_feat = [c for c in df_cancer.columns if c != 'target']
    
    with tab_pred:
        st.markdown('<div class="step-header">Step 1: Tumor Cell Measurements</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="helper-tip">
            💡 <b>Tip:</b> Slide the values below to match the cell biopsy readings. Default values represent typical averages.
        </div>
        """, unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            r_mean = st.slider("🔵 Cell Radius (Mean size in mm)", min_value=6.0, max_value=30.0, value=14.0, step=0.1,
                               help="Mean of distances from center to points on the perimeter.")
            t_mean = st.slider("🌫️ Cell Texture (Mean gray-scale variation)", min_value=9.0, max_value=40.0, value=19.0, step=0.1)
            p_mean = st.slider("⭕ Cell Perimeter (Mean circumference in mm)", min_value=40.0, max_value=190.0, value=90.0, step=0.5)
            a_mean = st.slider("📐 Cell Area (Mean size in sq mm)", min_value=140.0, max_value=2500.0, value=650.0, step=10.0)
            
        with c2:
            c_mean = st.slider("🕳️ Cell Concavity (Severity of contour indentations)", min_value=0.0, max_value=0.5, value=0.08, step=0.01)
            cp_mean = st.slider("📍 Concave Points (Number of concave portions)", min_value=0.0, max_value=0.25, value=0.05, step=0.005)
            r_worst = st.slider("🔴 Largest Cell Radius (Worst reading in mm)", min_value=7.0, max_value=36.0, value=16.0, step=0.1)
            cp_worst = st.slider("📌 Worst Concave Points Reading", min_value=0.0, max_value=0.30, value=0.11, step=0.005)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔬 Diagnose Tumor Status", type="primary", use_container_width=True):
            mean_vals = df_cancer[cancer_feat].mean().to_dict()
            mean_vals['radius_mean'] = r_mean
            mean_vals['texture_mean'] = t_mean
            mean_vals['perimeter_mean'] = p_mean
            mean_vals['area_mean'] = a_mean
            mean_vals['concavity_mean'] = c_mean
            mean_vals['concave points_mean'] = cp_mean
            mean_vals['radius_worst'] = r_worst
            mean_vals['concave points_worst'] = cp_worst
            
            input_df = pd.DataFrame([mean_vals])[cancer_feat]
            saved = st.session_state.trained_models['cancer']
            scaled = saved['scaler'].transform(input_df)
            pred = saved['model'].predict(scaled)[0]
            
            prob = None
            if hasattr(saved['model'], "predict_proba"):
                prob = saved['model'].predict_proba(scaled)[0]
                
            if pred == 1:
                conf = f"Confidence: **{prob[1]*100:.1f}% Malignant**" if prob is not None else ""
                st.markdown(f"""
                <div class="result-box-warning">
                    <div class="result-title">🚨 Diagnostic Result</div>
                    <div class="result-value" style="color: #F87171;">Malignant (Cancerous)</div>
                    <div class="result-details">
                        The cell patterns strongly align with malignant tumor characteristics. {conf}<br>
                        <i>Immediate clinical evaluation and biopsy review is recommended.</i>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                conf = f"Confidence: **{prob[0]*100:.1f}% Benign**" if prob is not None else ""
                st.markdown(f"""
                <div class="result-box-success">
                    <div class="result-title">✅ Diagnostic Result</div>
                    <div class="result-value" style="color: #34D399;">Benign (Non-Cancerous / Safe)</div>
                    <div class="result-details">
                        The cell structure matches benign, non-harmful mass patterns. {conf}<br>
                        <i>Always follow up with your doctor for periodic routine checkups.</i>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with tab_train:
        st.markdown('<div class="step-header">Model Accuracy & Selection</div>', unsafe_allow_html=True)
        saved = st.session_state.trained_models.get('cancer', {})
        m1, m2 = st.columns(2)
        m1.info(f"**Current Algorithm:** {saved.get('algo', 'Random Forest')}")
        m2.success(f"**Test Accuracy:** {saved.get('test_acc', 0.96)*100:.1f}%")
        
        cancer_algo_sel = st.selectbox(
            "Change Algorithm",
            ["Random Forest Classifier", "Support Vector Classifier (SVC)", "Gradient Boosting", "Logistic Regression"],
            key="cancer_algo_sel"
        )
        if st.button("🚀 Retrain Cancer Classifier", key="btn_retrain_cancer"):
            with st.spinner("Retraining model..."):
                X = df_cancer[cancer_feat]
                Y = df_cancer['target']
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=42)
                
                if "Random Forest" in cancer_algo_sel:
                    clf = RandomForestClassifier(n_estimators=100, random_state=42)
                elif "SVC" in cancer_algo_sel:
                    clf = SVC(kernel='rbf', probability=True, random_state=42)
                elif "Gradient" in cancer_algo_sel:
                    clf = GradientBoostingClassifier(random_state=42)
                else:
                    clf = LogisticRegression(random_state=42)
                    
                clf.fit(X_train, Y_train)
                acc = accuracy_score(Y_test, clf.predict(X_test))
                st.session_state.trained_models['cancer'] = {
                    'model': clf,
                    'scaler': scaler,
                    'features': cancer_feat,
                    'test_acc': acc,
                    'algo': cancer_algo_sel
                }
                st.success(f"Trained successfully! New Accuracy: {acc*100:.2f}%")

    with tab_data:
        st.markdown('<div class="step-header">About the Dataset</div>', unsafe_allow_html=True)
        st.markdown("Trained on the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset (569 biopsy cases).")
        st.dataframe(df_cancer.head(10), use_container_width=True)
        
        fig = px.scatter(df_cancer, x="radius_mean", y="texture_mean", color="target",
                         title="Mean Radius vs Mean Texture (0: Benign, 1: Malignant)",
                         color_discrete_map={0: "#3B82F6", 1: "#EC4899"},
                         labels={"target": "Diagnosis (0: Benign, 1: Malignant)"},
                         template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# 3. QUIKR CAR PRICE ESTIMATOR
# ==============================================================================
elif app_choice == "🚗 Used Car Price Estimator":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🚗 Used Car Fair Price Calculator</div>
        <p class="hero-desc">Calculate the fair resale market value of any used car in India in seconds.</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab_pred, tab_train, tab_data = st.tabs(["🔮 Estimate Price (Simple)", "⚙️ Model & Training", "📊 Explore Market Trends"])
    df_car = get_car_data()
    
    with tab_pred:
        st.markdown('<div class="step-header">Step 1: Select Your Car Details</div>', unsafe_allow_html=True)
        st.markdown('<div class="step-desc">Pick the brand, model, manufacturing year, and odometer reading.</div>', unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            companies = sorted(df_car['company'].unique().tolist())
            selected_company = st.selectbox("🏷️ Car Brand / Manufacturer", companies, index=companies.index("Maruti") if "Maruti" in companies else 0)
            
            models_for_company = sorted(df_car[df_car['company'] == selected_company]['name'].unique().tolist())
            if not models_for_company:
                models_for_company = sorted(df_car['name'].unique().tolist())
            selected_model = st.selectbox("🚘 Car Model Variant", models_for_company)
            
            selected_fuel = st.selectbox("⛽ Fuel Type", sorted(df_car['fuel_type'].unique().tolist()))
            
        with c2:
            years = sorted(df_car['year'].unique().tolist(), reverse=True)
            selected_year = st.selectbox("📅 Year of Manufacture", years, index=min(4, len(years)-1))
            
            selected_kms = st.slider("🛣️ Total Kilometers Driven", min_value=1000, max_value=250000, value=38000, step=1000,
                                     help="Approximate odometer reading on the car.")
            
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("💰 Calculate Fair Resale Price", type="primary", use_container_width=True):
            query_df = pd.DataFrame([[selected_model, selected_company, selected_year, selected_kms, selected_fuel]],
                                    columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
            
            pipeline = st.session_state.trained_models['car']['pipeline']
            est_price = pipeline.predict(query_df)[0]
            if est_price < 25000:
                est_price = 25000  # Minimum realistic salvage floor
                
            min_range = int(est_price * 0.93)
            max_range = int(est_price * 1.07)
            
            st.markdown(f"""
            <div class="result-box-price">
                <div class="result-title">🏷️ Estimated Fair Market Value</div>
                <div class="result-value" style="color: #818CF8;">₹ {int(est_price):,}</div>
                <div class="result-details">
                    <b>Expected Price Range:</b> ₹ {min_range:,} – ₹ {max_range:,}<br>
                    <span style="color: #94A3B8; font-size: 0.85rem; margin-top: 4px; display: inline-block;">
                        Vehicle: {selected_model} ({selected_company}) • Year: {selected_year} • {selected_kms:,} km • {selected_fuel}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab_train:
        st.markdown('<div class="step-header">Price Pipeline & Training</div>', unsafe_allow_html=True)
        saved = st.session_state.trained_models.get('car', {})
        m1, m2, m3 = st.columns(3)
        m1.info(f"**Pipeline:** {saved.get('algo', 'Linear Regression')}")
        m2.success(f"**R² Accuracy Score:** {saved.get('r2', 0.76):.2f}")
        m3.metric("Average Error (MAE)", f"₹ {int(saved.get('mae', 85000)):,}")
        
        car_algo_sel = st.selectbox(
            "Change Regression Model",
            ["Linear Regression", "Ridge Regression", "Random Forest Regressor", "Gradient Boosting Regressor"],
            key="car_algo_sel"
        )
        if st.button("🚀 Retrain Car Valuation Model", key="btn_retrain_car"):
            with st.spinner("Retraining car price model..."):
                X = df_car[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
                y = df_car['Price']
                ohe = OneHotEncoder()
                ohe.fit(X[['name', 'company', 'fuel_type']])
                col_trans = make_column_transformer(
                    (OneHotEncoder(categories=ohe.categories_), ['name', 'company', 'fuel_type']),
                    remainder='passthrough'
                )
                if "Linear" in car_algo_sel:
                    reg = LinearRegression()
                elif "Ridge" in car_algo_sel:
                    reg = Ridge(alpha=1.0)
                elif "Random Forest" in car_algo_sel:
                    reg = RandomForestRegressor(n_estimators=100, random_state=42)
                else:
                    reg = GradientBoostingRegressor(random_state=42)
                    
                pipe = make_pipeline(col_trans, reg)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                st.session_state.trained_models['car'] = {
                    'pipeline': pipe,
                    'r2': r2,
                    'mae': mae,
                    'algo': car_algo_sel
                }
                st.success(f"Retrained! R² Score: {r2:.3f}, Average Error: ₹{mae:,.0f}")

    with tab_data:
        st.markdown('<div class="step-header">Car Market Overview</div>', unsafe_allow_html=True)
        st.markdown(f"Trained on **{len(df_car)} verified used car listings** from Quikr.")
        st.dataframe(df_car.head(10), use_container_width=True)
        
        fig = px.box(df_car[df_car['Price'] < 2000000], x="company", y="Price",
                     title="Resale Price Distributions across Top Car Brands (Under 20 Lakhs)",
                     template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
