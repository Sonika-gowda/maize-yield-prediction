import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap

# PAGE CONFIG

st.set_page_config(page_title="Maize Yield Prediction", page_icon="🌽", layout="wide")

def set_bg():
    st.markdown("""
    <style>

    /* Background image */
    /* Background image */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(
        rgba(0, 0, 0, 0.55),
        rgba(0, 0, 0, 0.55)
    ),
    url("https://images.pexels.com/photos/20234940/pexels-photo-20234940/free-photo-of-a-corn-field.jpeg");
    
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

    /* Dark overlay for readability */
    [data-testid="stAppViewContainer"] > .main {
        background-color: rgba(0,0,0,0.55);
    }

    /* Glass container */
    .glass {
        background: rgba(0,0,0,0.65);
        padding: 20px;
        border-radius: 15px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        margin-bottom: 20px;
    }

    /* Text color fix */
    h1, h2, h3, h4, p, label {
        color: white !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(0,0,0,0.8);
    }

    /* Buttons */
    .stButton>button {
        background-color: #16a34a;
        color: white;
        border-radius: 10px;
    }

    </style>
    """, unsafe_allow_html=True)

set_bg()
st.markdown("""
<div class="glass">
<h1 style='text-align:center;'>🌽 Smart Maize Yield Predictor</h1>
<p style='text-align:center;'>AI-powered Agricultural Prediction System</p>
</div>
""", unsafe_allow_html=True)
# LOAD MODEL

model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

df = pd.read_csv("archive/yield_df.csv")
maize_df = df[df["Item"] == "Maize"]
average_yield = maize_df["hg/ha_yield"].mean()

# SIDEBAR

st.sidebar.title("🌽 Navigation")

page = st.sidebar.radio(
    "Go To",
    [
    "Home",
    "Dataset Overview",
    "Prediction",
    "Rainfall Analysis",
    "Feature Importance",
    "SHAP Explainability",
    "Model Performance"
]
)
# HOME PAGE

if page == "Home":
    
    st.title("🌽 Maize Yield Prediction Under Rainfall Variability")

    st.markdown("""
    <div class="glass">
    

    Predict maize yield based on:

    - Rainfall
    - Temperature
    - Pesticide Usage
    - Year
    - Area (Country)

    using Decision Tree Regression and Explainable AI.
    """, unsafe_allow_html=True)
    
    st.markdown("### 📊 System Status")

    st.info("Model: Trained Random Forest / ML Model")
    st.info("Status: Active ✅")
   
# DATASET PAGE

elif page == "Dataset Overview":

    
    st.title("📊 Dataset Overview")

    st.metric("Total Records", len(maize_df))
    st.metric("Countries", maize_df["Area"].nunique())
    st.metric("Year Range",
          f"{maize_df['Year'].min()} - {maize_df['Year'].max()}")
    
    st.subheader("Yield Distribution")

    fig, ax = plt.subplots()

    ax.hist(
        maize_df["hg/ha_yield"],
        bins=30
    )

    ax.set_xlabel("Yield (hg/ha)")
    ax.set_ylabel("Frequency")

    st.pyplot(fig)

    st.subheader("Rainfall Distribution")

    fig2, ax2 = plt.subplots()

    ax2.hist(
    maize_df["average_rain_fall_mm_per_year"],
    bins=30
    )

    ax2.set_xlabel("Rainfall (mm/year)")
    ax2.set_ylabel("Frequency")

    st.pyplot(fig2)

    st.subheader("Top 10 Countries by Average Maize Yield")
    country_yield = maize_df.groupby("Area")[
    "hg/ha_yield"
    ].mean().sort_values(
    ascending=False
    ).head(10)

    fig, ax = plt.subplots()

    country_yield.plot(
    kind="bar",
    ax=ax
    )

    ax.set_ylabel("Average Yield")

    st.pyplot(fig)
     
    st.subheader("Sample Data")

    st.dataframe(maize_df.head())

  
# PREDICTION PAGE

elif page == "Prediction":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.title("🔮 Predict Maize Yield")

    countries = sorted(maize_df["Area"].unique())

    area = st.selectbox(
        "Select Country",
        countries
    )

    year = st.number_input(
        "Year",
        min_value=1990,
        max_value=2035,
        value=2020
    )

    rainfall = st.number_input(
        "Average Rainfall (mm/year)",
        value=1000.0
    )

    pesticides = st.number_input(
        "Pesticides (tonnes)",
        value=100.0
    )

    temp = st.number_input(
        "Average Temperature",
        value=20.0
    )

    if st.button("Predict Yield"):

        area_encoded = label_encoder.transform([area])[0]

        input_data = pd.DataFrame(
            [[
                area_encoded,
                year,
                rainfall,
                pesticides,
                temp
            ]],
            columns=[
                'Area',
                'Year',
                'average_rain_fall_mm_per_year',
                'pesticides_tonnes',
                'avg_temp'
            ]
        )

        prediction = model.predict(input_data)
        predicted_yield = prediction[0]

        st.success(
            f"🌽 Predicted Yield: {predicted_yield:,.2f} hg/ha"
        )

        # =============================
        # Prediction Interpretation
        # =============================

        st.subheader("📊 Prediction Interpretation")

        st.write(
            f"Average Maize Yield in Dataset: **{average_yield:,.2f} hg/ha**"
        )

        difference = predicted_yield - average_yield

        if difference > 0:
            st.success(
                f"The predicted yield is **{difference:,.2f} hg/ha above** the dataset average."
            )
        else:
            st.warning(
                f"The predicted yield is **{abs(difference):,.2f} hg/ha below** the dataset average."
            )
 # Why Prediction
        
        st.subheader("🧠 Why did the model predict this value?")

        if rainfall > 1200:
            st.write(
                "✔ High rainfall generally supports better maize growth."
            )

        elif rainfall < 700:
            st.write(
                "⚠ Low rainfall can negatively affect maize production."
            )

        else:
            st.write(
                "✔ Rainfall is within a moderate range."
            )

        if 20 <= temp <= 30:
            st.write(
                "✔ Temperature is within the ideal range for maize cultivation."
            )
        else:
            st.write(
                "⚠ Temperature is outside the optimal range for maize growth."
            )

        if pesticides > 500:
            st.write(
                "✔ Higher pesticide usage may reduce crop damage."
            )
        else:
            st.write(
                "✔ Pesticide usage is within a normal range."
            )

        if year > 2010:
            st.write(
                "✔ Modern farming practices may contribute to higher yields."
            )
  # Recommendations
       
        st.subheader("💡 Recommendations")

        recommendations = []

        if rainfall < 700:
            recommendations.append(
                "⚠ Low rainfall detected. Irrigation support is recommended."
            )

        elif rainfall > 1200:
            recommendations.append(
                "✔ Rainfall conditions appear favorable for maize cultivation."
            )

        else:
            recommendations.append(
                "✔ Rainfall is within a moderate range for maize production."
            )

        if temp > 35:
            recommendations.append(
                "⚠ Very high temperature may stress maize crops."
            )

        elif 20 <= temp <= 30:
            recommendations.append(
                "✔ Temperature is suitable for healthy maize growth."
            )

        else:
            recommendations.append(
                "⚠ Temperature is slightly outside the optimal maize range."
            )

        if pesticides < 50:
            recommendations.append(
                "⚠ Low pesticide usage may increase pest-related losses."
            )

        else:
            recommendations.append(
                "✔ Pesticide usage appears adequate."
            )

        for rec in recommendations:
            st.write(rec)

    st.markdown('</div>', unsafe_allow_html=True)
elif page == "Rainfall Analysis":
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.title("🌧 Rainfall Analysis")

    st.subheader("Rainfall vs Yield")

    fig, ax = plt.subplots()

    ax.scatter(
        maize_df["average_rain_fall_mm_per_year"],
        maize_df["hg/ha_yield"]
    )

    ax.set_xlabel("Rainfall")
    ax.set_ylabel("Yield")

    st.pyplot(fig)

    st.subheader("Temperature vs Yield")

    fig2, ax2 = plt.subplots()

    ax2.scatter(
        maize_df["avg_temp"],
        maize_df["hg/ha_yield"]
    )

    ax2.set_xlabel("Temperature")
    ax2.set_ylabel("Yield")

    st.pyplot(fig2)
    st.markdown('</div>', unsafe_allow_html=True)
elif page == "Feature Importance":
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)

    st.title("📈 Feature Importance")

    feature_names = [
        "Area",
        "Year",
        "Rainfall",
        "Pesticides",
        "Temperature"
    ]

    importance = model.feature_importances_

    fig, ax = plt.subplots()

    ax.barh(
        feature_names,
        importance
    )

    ax.set_xlabel("Importance")

    st.pyplot(fig)


elif page == "SHAP Explainability":

    st.title("🔍 SHAP Explainability")

    st.write(
        "Shows how features influence maize yield predictions."
    )

    sample = maize_df.drop(
        columns=["Item", "Unnamed: 0"],
        errors="ignore"
    )

    sample["Area"] = label_encoder.transform(
        sample["Area"]
    )

    X_sample = sample.drop(
        "hg/ha_yield",
        axis=1
    )

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_sample)

    fig = plt.figure()

    shap.summary_plot(
        shap_values,
        X_sample,
        show=False
    )

    st.pyplot(plt.gcf())
    st.markdown('</div>', unsafe_allow_html=True)
# PERFORMANCE PAGE

elif page == "Model Performance":
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)

    st.title("📈 Model Performance")

    st.metric("Best Max Depth", "15")
    st.metric("MAE", "3061.01")
    st.metric("RMSE", "6834.06")
    st.metric("R² Score", "0.9364")

    st.success(
        "Model explains approximately 93.64% of yield variation."
    )  
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
<hr style="border:1px solid #444;">
<p style='text-align:center; color:gray;'>
🌽 Smart Agriculture AI Project | Built using Streamlit<br>
© 2026 All Rights Reserved
</p>
""", unsafe_allow_html=True)