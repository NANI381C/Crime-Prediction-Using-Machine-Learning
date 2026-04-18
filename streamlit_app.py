import sys
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import folium
import plotly.express as px
from folium.plugins import TimestampedGeoJson

# ---------------------------------------------
# IMPORT PROJECT MODULES
# ---------------------------------------------
from data_prep import load_and_preprocess
from model_lstm import train_and_forecast
from hotspot_cluster import generate_hotspot_map, generate_risk_map
from shap_explainer import explain_lstm_with_shap
from classifier_model import train_classifier
from visuals_map import build_city_geo_df, create_timestamped_geojson


# ---------------------------------------------
# STREAMLIT CONFIG
# ---------------------------------------------
st.set_page_config(page_title="Crime Forecasting Dashboard",
                   layout="wide", page_icon="🔍")

# ---------------------------------------------
# DARK UI CSS
# ---------------------------------------------
st.markdown("""
<style>
:root {
    --bg: #0b0f14;
    --card: rgba(255,255,255,0.03);
    --accent: #34eaff;
    --muted: #9aa5b1;
}
.stApp { background: var(--bg); color: white; }
.card {
    background: var(--card);
    padding: 16px;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 6px 18px rgba(0,0,0,0.6);
}
.metric {
    color: var(--accent);
    font-size: 34px;
    font-weight:800;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------
# HEADER
# ---------------------------------------------
st.markdown("<h1 style='color:white;'>🔍 Crime Forecasting Dashboard</h1>",
            unsafe_allow_html=True)

st.write("LSTM Forecasting • Hotspot Mapping • Google Maps • Classifier • SHAP Explainability")


# ---------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------
st.sidebar.title("⚙️ Controls")
uploaded = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"])

forecast_horizon = st.sidebar.slider("Forecast Steps (Months)", 1, 12, 3)
epochs = st.sidebar.slider("Training Epochs", 10, 200, 60)
window = st.sidebar.slider("LSTM Window", 1, 12, 3)

use_google_maps = st.sidebar.checkbox("Use Google Maps Tiles", True)

GOOGLE_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", None)
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_MAPS_API_KEY"]
except:
    pass


# ---------------------------------------------
# LOAD DATASET
# ---------------------------------------------
df_monthly = None
encoders = {}

if uploaded:
    try:
        df_monthly, encoders = load_and_preprocess(uploaded)
        st.success("Dataset Loaded Successfully 🎉")
    except Exception as e:
        st.error("Failed to load dataset:")
        st.error(str(e))

else:
    # fallback if no upload
    if os.path.exists("data/time_series_city.csv"):
        df_monthly = pd.read_csv("data/time_series_city.csv")
        st.info("Loaded default prepared dataset.")
    else:
        st.warning("Upload a dataset to start.")
        df_monthly = None


# ---------------------------------------------
# FIX: CLEAN DATE COLUMN IF NEEDED
# ---------------------------------------------
if df_monthly is not None:
    if "Year" in df_monthly.columns and "Month" in df_monthly.columns:
        try:
            df_monthly["Year"] = df_monthly["Year"].astype(int)
            df_monthly["Month"] = df_monthly["Month"].astype(int)
            df_monthly["Date"] = pd.to_datetime(
                df_monthly["Year"].astype(str)
                + "-" +
                df_monthly["Month"].astype(str).str.zfill(2)
                + "-01",
                errors="coerce"
            )
        except:
            pass


# ---------------------------------------------
# TABS
# ---------------------------------------------
tabs = st.tabs([
    "📊 Forecasting",
    "🗺 Hotspots",
    "📄 Project Info",
    "🧠 Classification",
    "📈 Multi-City",
    "🎞 Animated Heatmap"
])


# ===========================================================
# TAB 1 — FORECASTING
# ===========================================================
with tabs[0]:

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("📊 LSTM Crime Forecasting")

    if df_monthly is None:
        st.warning("Upload a dataset first.")
    else:

        # ---------------------------------------------
        # FIX: Load City Mapping (Encoded → Name)
        # ---------------------------------------------
        city_map_path = os.path.join(ROOT_DIR, "models", "city_mapping.csv")
        alt_city_map_path = os.path.join(ROOT_DIR, "city_mapping.csv")
        if os.path.exists(city_map_path):
            mapping_path = city_map_path
        elif os.path.exists(alt_city_map_path):
            mapping_path = alt_city_map_path
        else:
            mapping_path = None

        city_col = None
        for col in ["City", "city", "CITY", "City_enc"]:
            if col in df_monthly.columns:
                city_col = col
                break

        if city_col is None:
            st.error("City column not found in dataset.")
        else:

            # ---------------------------------------------
            # If mapping exists, convert numbers to names
            # ---------------------------------------------
            if mapping_path is not None:
                st.success("City mapping loaded successfully ✔")
                mapping = pd.read_csv(mapping_path)
                mapping = mapping.rename(columns={
                    mapping.columns[0]: "CityName",
                    mapping.columns[1]: "CityCode"
                })

                # If df contains encoded numbers like 0,1,2...
                if df_monthly[city_col].dtype != object:

                    # merge encoded → city name
                    df_monthly = df_monthly.merge(
                        mapping,
                        left_on=city_col,
                        right_on="CityCode",
                        how="left"
                    )

                    city_display_col = "CityName"
                else:
                    # already names
                    city_display_col = city_col

            else:
                st.error("City mapping file missing: models/city_mapping.csv")
                city_display_col = city_col

            # ---------------------------------------------
            # Build city dropdown with names instead of numbers
            # ---------------------------------------------
            cities = sorted(df_monthly[city_display_col].dropna().unique())
            city_choice = st.selectbox("Select City", cities)

            # ---------------------------------------------
            # Convert chosen name → encoded for processing
            # ---------------------------------------------
            if mapping_path is not None:
                selected_code = int(mapping[mapping["CityName"] == city_choice]["CityCode"].values[0])
                df_city = df_monthly[df_monthly["CityCode"] == selected_code].sort_values("Date")
            else:
                df_city = df_monthly[df_monthly[city_display_col] == city_choice].sort_values("Date")

            st.write("Recent rows:")
            st.dataframe(df_city.tail(10))

            # ---------------------------------------------
            # Train & Forecast
            # ---------------------------------------------
            if st.button("🚀 Train & Forecast"):
                with st.spinner("Training LSTM..."):
                    model, scaler_X, scaler_y, metrics, future_preds = train_and_forecast(
                        df_city,
                        window=window,
                        epochs=epochs,
                        forecast_steps=forecast_horizon
                    )

                st.success("Training complete!")

                accuracy = 100 - metrics["mape"]

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy", f"{accuracy:.2f}%")
                c2.metric("MAPE", f"{metrics['mape']:.2f}%")
                c3.metric("RMSE", f"{metrics['rmse']:.2f}")
                c4.metric("MAE", f"{metrics['mae']:.2f}")

                # Recent performance chart
                actual_df = pd.DataFrame({
                    "Date": df_city["Date"].iloc[-len(metrics["test_true"]):].reset_index(drop=True),
                    "Actual": metrics["test_true"],
                    "Predicted": metrics["test_pred"]
                })
                fig_history = px.line(
                    actual_df,
                    x="Date",
                    y=["Actual", "Predicted"],
                    title="Recent Actual vs Predicted",
                    labels={"value": "Crime Count", "variable": "Series"}
                )
                fig_history.update_layout(template="plotly_dark")
                st.plotly_chart(fig_history, use_container_width=True)

                # Forecast chart
                st.subheader("🔮 Future Forecast")
                last_date = df_city["Date"].max() if "Date" in df_city.columns else None
                if last_date is not None and not pd.isna(last_date):
                    future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(len(future_preds))]
                else:
                    future_dates = [f"Month +{i+1}" for i in range(len(future_preds))]

                forecast_df = pd.DataFrame({
                    "Period": future_dates,
                    "Predicted_Crimes": [float(p) for p in future_preds]
                })

                fig_forecast = px.area(
                    forecast_df,
                    x="Period",
                    y="Predicted_Crimes",
                    title="Crime Forecast Trend",
                    markers=True
                )
                fig_forecast.update_layout(
                    xaxis_title="Period",
                    yaxis_title="Predicted Crime Count",
                    template="plotly_dark"
                )
                st.plotly_chart(fig_forecast, use_container_width=True)

                avg_pred = forecast_df["Predicted_Crimes"].mean()
                max_row = forecast_df.loc[forecast_df["Predicted_Crimes"].idxmax()]
                max_period = max_row["Period"]
                max_period_str = max_period.strftime("%b %Y") if isinstance(max_period, pd.Timestamp) else str(max_period)
                trend = "increasing" if forecast_df["Predicted_Crimes"].iloc[-1] > forecast_df["Predicted_Crimes"].iloc[0] else "decreasing"

                st.subheader("🧠 AI Insights")
                st.info(
                    f"🔎 AI Analysis\n\n"
                    f"• Average predicted crimes: {avg_pred:.1f}\n"
                    f"• Highest crime expected in: {max_period_str}\n"
                    f"• Trend: {trend.title()}\n\n"
                    "Recommendation: Increase surveillance during high-risk months."
                )

                st.markdown("#### Forecast Details")
                st.dataframe(forecast_df)

                # Save model
                os.makedirs("models", exist_ok=True)
                try:
                    model.save("models/crime_lstm_model.keras")
                except Exception:
                    joblib.dump(model, "models/crime_lstm_model.joblib")

                joblib.dump(scaler_X, "models/scaler_X.joblib")
                joblib.dump(scaler_y, "models/scaler_y.joblib")

    st.markdown("</div>", unsafe_allow_html=True)


# ===========================================================
# TAB 2 — HOTSPOTS
# ===========================================================
with tabs[1]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("🗺 Crime Hotspot Map")

    if df_monthly is not None:
        if st.button("Generate Hotspot Map"):

            with st.spinner("Generating map..."):

                m = generate_hotspot_map(df_monthly, encoders)

                if m is None:
                    st.error("Unable to generate hotspot map. No valid location data found.")
                else:
                    # Optional: Replace tiles with Google Maps
                    if use_google_maps and GOOGLE_API_KEY:
                        folium.TileLayer(
                            tiles=f"https://mt1.google.com/vt/lyrs=m&x={{x}}&y={{y}}&z={{z}}&key={GOOGLE_API_KEY}",
                            attr="Google",
                            name="Google Maps"
                        ).add_to(m)
                        m.add_child(folium.LayerControl())

                    m.save("map.html")

                    city_label = None
                    for col in ["CityName", "City", "city", "CITY", "City_enc"]:
                        if col in df_monthly.columns:
                            city_label = col
                            break
                    if city_label is None:
                        city_label = "City_enc"

                    top_cities = df_monthly.groupby(city_label)["Crime_Count"].sum().sort_values(ascending=False).head(10)

                    map_col, rank_col = st.columns([2, 1])
                    with map_col:
                        st.iframe("map.html", height=620)
                    with rank_col:
                        st.subheader("Top 10 Crime Hotspots")
                        st.bar_chart(top_cities)
                        st.write(top_cities.reset_index().rename(columns={city_label: "City", "Crime_Count": "Total Crimes"}))

        if st.button("🚨 Generate Live Crime Risk Map"):
            with st.spinner("Generating live risk map..."):
                risk_map = generate_risk_map(df_monthly, encoders)
                if risk_map is None:
                    st.error("Unable to generate live risk map. No valid city locations found.")
                else:
                    risk_map.save("risk_map.html")
                    st.iframe("risk_map.html", height=620)
                    st.markdown(
                        "### Risk Level Guide\n\n"
                        "🟢 Low Risk  \n"
                        "🟠 Medium Risk  \n"
                        "🔴 High Risk"
                    )

    st.markdown("</div>", unsafe_allow_html=True)


# ===========================================================
# TAB 3 — PROJECT INFO
# ===========================================================
with tabs[2]:

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("📄 Project Overview")

    st.write("""
    **Key Features**
    - LSTM Time-Series Crime Forecasting  
    - SHAP Explainability  
    - Hotspot Detection  
    - Crime Classification  
    - Multi-City Trend Comparison  
    - Animated Crime Maps  
    """)

    st.markdown("</div>", unsafe_allow_html=True)


# ===========================================================
# TAB 4 — CLASSIFICATION
# ===========================================================
with tabs[3]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("🧠 Crime Classification")

    if uploaded:
        if st.button("Train Classifier"):
            uploaded.seek(0)
            raw = pd.read_csv(uploaded)

            with st.spinner("Training..."):
                metrics = train_classifier(raw)

            st.success("Classifier Trained!")
            st.write(metrics["accuracy"])
            st.json(metrics["report"])

    st.markdown("</div>", unsafe_allow_html=True)


# ===========================================================
# TAB 5 — MULTI CITY COMPARISON
# ===========================================================
with tabs[4]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("📈 Multi-City Comparison")

    if df_monthly is not None:

        city_col = "City" if "City" in df_monthly.columns else df_monthly.columns[0]
        cities = sorted(df_monthly[city_col].unique())

        chosen = st.multiselect("Select Cities", cities, default=cities[:3])

        if chosen:
            df_sel = df_monthly[df_monthly[city_col].isin(chosen)]
            pivot = df_sel.pivot_table(index="Date",
                                       columns=city_col,
                                       values="Crime_Count")
            st.line_chart(pivot)

    st.markdown("</div>", unsafe_allow_html=True)


# ===========================================================
# TAB 6 — ANIMATED HEATMAP
# ===========================================================
with tabs[5]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("🎞 Animated Crime Heatmap")

    if df_monthly is not None:
        if st.button("Generate Animated Heatmap"):
            with st.spinner("Generating Map..."):
                geo = build_city_geo_df(df_monthly, encoders)
                geojson = create_timestamped_geojson(df_monthly, geo)

                m = folium.Map(location=[20.5937, 78.9629],
                               zoom_start=5,
                               tiles="CartoDB dark_matter")

                TimestampedGeoJson(geojson,
                                   transition_time=700,
                                   auto_play=True).add_to(m)

                m.save("animated_map.html")
                st.iframe("animated_map.html", height=650)

    st.markdown("</div>", unsafe_allow_html=True)


# END OF FILE
