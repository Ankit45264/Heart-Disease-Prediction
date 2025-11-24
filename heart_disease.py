import pickle
import numpy as np
import streamlit as st

st.title("Heart Disease Prediction Using Machine Learning")
st.image('https://thumbs.dreamstime.com/b/experience-essence-life-vivid-d-human-heart-pulsating-dynamic-ekg-line-perfect-medical-health-395635056.jpg')
st.sidebar.image('https://static.vecteezy.com/system/resources/thumbnails/023/560/057/small_2x/background-with-a-heart-with-the-heartbeat-monitor-line-heart-and-heartbeat-symbol-generative-ai-photo.jpg')

page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background:
        linear-gradient(rgba(0,0,0,0.8), rgba(0,0,0,0.8)),
        url("https://i.pinimg.com/originals/76/d2/9a/76d29aa5cadeff7ff8881ef349be0a3f.gif");
    background-size: 270px 270px;
    background-repeat: repeat;
    background-attachment: fixed;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Sidebar Content
st.sidebar.title("Input Parameters")

age = st.sidebar.slider("Select age value", 29, 77, 50)
trestbps = st.sidebar.slider("Resting Blood Pressure (trestbps)", 94, 200, 120)
chol = st.sidebar.slider("Serum Cholestoral (chol)", 126, 564, 200)
thalach = st.sidebar.slider("Max Heart Rate (thalach)", 71, 202, 150)
oldpeak = st.sidebar.slider("ST depression (oldpeak)", 0.0, 6.0, 2.5)

sex = st.sidebar.radio(
    "Sex",
    options=[0, 1],
    format_func=lambda x: "Female" if x == 0 else "Male"
)

cp = st.sidebar.radio(
    "Chest Pain Type (cp)",
    options=[0, 1, 2, 3],
    format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"][x]
)

fbs = st.sidebar.radio(
    "Fasting Blood Sugar > 120 mg/dl (fbs)",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

restecg = st.sidebar.radio(
    "Resting ECG Results (restecg)",
    options=[0, 1, 2],
    format_func=lambda x: ["Normal", "ST-T abnormality", "LV Hypertrophy"][x]
)

exang = st.sidebar.radio(
    "Exercise Induced Angina (exang)",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

slope = st.sidebar.radio(
    "Slope of Peak Exercise ST Segment (slope)",
    options=[0, 1, 2],
    format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x]
)

ca = st.sidebar.radio(
    "Number of Major Vessels (ca)",
    options=[0, 1, 2, 3, 4],
    format_func=lambda x: f"{x} vessel(s)"
)

thal = st.sidebar.radio(
    "Thalassemia (thal)",
    options=[0, 1, 2, 3],
    format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect", "Unknown"][x]
)

st.subheader("Predicted Output")

# Robust model loading
def load_model(pkl_path):
    with open(pkl_path, 'rb') as f:
        loaded = pickle.load(f)
    # If it's a dict/tuple try to locate the estimator
    if hasattr(loaded, "predict"):
        return loaded
    if isinstance(loaded, dict):
        # common keys that might store the estimator
        for key in ("model", "estimator", "pipeline", "clf"):
            if key in loaded and hasattr(loaded[key], "predict"):
                return loaded[key]
        # sometimes sklearn pipeline stored under 'pipeline' or 'pipe'
        for key in loaded:
            if hasattr(loaded[key], "predict"):
                return loaded[key]
    if isinstance(loaded, (list, tuple)):
        # try to find first element with predict
        for item in loaded:
            if hasattr(item, "predict"):
                return item
    # nothing found
    return loaded

try:
    model = load_model('heart_disease_prediction.pkl')
except FileNotFoundError:
    st.error("Pickle file 'heart_disease_prediction.pkl' not found. Put it in the same directory as this script.")
    model = None
except Exception as e:
    st.error(f"Error loading model pickle: {e}")
    model = None

if st.sidebar.button("Predict"):
    # prepare input
    data1 = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                       thalach, exang, oldpeak, slope, ca, thal]])
    if model is None:
        st.error("Model is not loaded. Prediction is not possible.")
    else:
        # check that model has predict
        if not hasattr(model, "predict"):
            st.error("Loaded object does not appear to be a model/pipeline with a .predict() method. "
                     "Check the contents of 'heart_disease_prediction.pkl'.")
        else:
            try:
                with st.spinner("Predicting..."):
                    prediction = model.predict(data1)[0]
                if int(prediction) == 0:
                    st.success("✔️🎉 No Heart Disease Detected ❤️💚")
                else:
                    st.error("⚠️ Heart Disease Detected 💔🩺")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

st.write("---")
st.write("Developed by **Ankit Kumar Maurya**")
