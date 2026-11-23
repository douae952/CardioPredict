import streamlit as st
import pandas as pd

try:
    import joblib
except ModuleNotFoundError:
    st.error("La bibliothèque joblib est absente. Installez-la avec : pip install joblib")
    st.stop()

st.set_page_config(
    page_title="CardioPredict",
    page_icon="🫀",
    layout="wide"
)

# Charger le modèle
saved = joblib.load("xgb_cardio.pkl")
model = saved["model"]
features = saved["features"]

st.title("🫀 CardioPredict")
st.subheader("Prédiction des maladies cardiovasculaires")
st.write("Modèle utilisé : XGBoost")

st.divider()

st.header("Informations du patient")

with st.form("cardio_form"):

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Âge (années)", 18, 100, 50
        )

        gender = st.selectbox(
            "Sexe", [1, 2],
            format_func=lambda x: "Femme" if x == 1 else "Homme"
        )

        bmi = st.number_input(
            "BMI", 10.0, 80.0, 25.0
        )

        ap_hi = st.number_input(
            "Pression systolique", 70, 250, 120
        )

        ap_lo = st.number_input(
            "Pression diastolique", 40, 180, 80
        )

    with col2:
        cholesterol = st.selectbox(
            "Cholestérol",
            [1, 2, 3],
            format_func=lambda x: {
                1: "Normal",
                2: "Au-dessus de la normale",
                3: "Bien au-dessus de la normale"
            }[x]
        )

        gluc = st.selectbox(
            "Glucose",
            [1, 2, 3],
            format_func=lambda x: {
                1: "Normal",
                2: "Au-dessus de la normale",
                3: "Bien au-dessus de la normale"
            }[x]
        )

        smoke = st.selectbox(
            "Fumeur", [0, 1],
            format_func=lambda x: "Non" if x == 0 else "Oui"
        )

        alco = st.selectbox(
            "Consommation d'alcool", [0, 1],
            format_func=lambda x: "Non" if x == 0 else "Oui"
        )

        active = st.selectbox(
            "Activité physique", [0, 1],
            format_func=lambda x: "Non" if x == 0 else "Oui"
        )

    submitted = st.form_submit_button(
        "Effectuer la prédiction",
        use_container_width=True
    )

if submitted:

    patient = {
        "age_in_years": age,
        "gender": gender,
        "bmi": bmi,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": cholesterol,
        "gluc": gluc,
        "smoke": smoke,
        "alco": alco,
        "active": active
    }

    X_new = pd.DataFrame([patient])[features]

    prediction = model.predict(X_new)[0]
    probability = model.predict_proba(X_new)[0][1]

    st.divider()
    st.header("Résultat")

    if prediction == 1:
        st.error("Classe prédite : 1")
        st.write("Le modèle prédit la présence d'une maladie cardiovasculaire.")
    else:
        st.success("Classe prédite : 0")
        st.write("Le modèle prédit l'absence de maladie cardiovasculaire.")

    st.metric(
        "Probabilité estimée de la classe 1",
        f"{probability * 100:.2f}%"
    )

    
