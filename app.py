import pandas as pd
import numpy as np
import joblib
import streamlit as st

@st.cache_resource
def main():
    pass

def load_model(path):
    artifact = joblib.load(path)
    return artifact["model"], artifact["features"], artifact["pandas_version"]

def build_features(surface, city, housing_unit, floor, num_rooms  ):
    # surface = float(input("Inserisci superfice: " ))
    # city = str(input("Inserisci città: "))
    # housing_unit = str(input("Inserisci tipo di abitazione: "))
    # floor = float(input("A quale piano si trova? ")) 
    # num_rooms = float(input("Quante stanze ha? "))

    # print(surface)
    # print(city)
    # print(housing_unit)
    # print(floor)
    # print(num_rooms)

    df_predict = pd.DataFrame(columns=['Surface', 'City', 'Housing_unit', 'floor',  'num_rooms'])

    df_predict.loc[len(df_predict)] = {
        'Surface' : surface,
        'City': city,
        'Housing_unit': housing_unit,
        'floor': floor,
        'num_rooms':num_rooms
    }

    city_size = {
        "Roma": "big",
        "Milano": "big",
        "Napoli": "big",
        "Torino": "mid-big",
        "Palermo": "mid-big",
        "Genova": "mid-big",
        "Bologna": "mid",
        "Firenze": "mid",
        "Bari": "mid",
        "Catania": "mid",
        "Verona": "mid",
        "Venezia": "mid",
        "Messina": "mid",
        "Padova": "mid",
        "Trieste": "mid",
        "Brescia": "mid",
        "Taranto": "mid",
        "Prato": "mid",
        "Parma": "mid-small",
        "Modena": "mid-small"
    }

    df_predict["city_size"] = df_predict["City"].map(city_size)

    #######################################################################

    city_macroregion = {
        "Milano": "north",
        "Torino": "north",
        "Genova": "north",
        "Bologna": "north",
        "Verona": "north",
        "Venezia": "north",
        "Padova": "north",
        "Trieste": "north",
        "Parma": "north",
        "Brescia": "north",
        "Modena": "north",
        "Prato": "center",
        "Roma": "center",
        "Firenze": "center",
        "Napoli": "south",
        "Palermo": "south",
        "Bari": "south",
        "Catania": "south",
        "Messina": "south",
        "Taranto": "south"
    }

    df_predict["macroregion"] = df_predict["City"].map(city_macroregion)

    #######################################################################

    bins = [-np.inf, 40, 70, 100, 200, 400, np.inf]
    labels = [
        "very small",
        "small",
        "mid-small",
        "mid",
        "large",
        "very large"
    ]

    df_predict["surface_bracket"] = pd.cut(
        df_predict["Surface"],
        bins=bins,
        labels=labels,
        right=False
    )

    #######################################################################

    df_predict = df_predict[['Surface', 'City', 'Housing_unit', 'city_size', 'macroregion', 'floor',
        'num_rooms', 'surface_bracket']]
    
    df_predict[['City', 'Housing_unit', 'city_size', 
                   'macroregion', 'surface_bracket']] = df_predict[['City', 'Housing_unit', 'city_size', 'macroregion', 'surface_bracket']].astype('category')
    
    # print(df_predict)
    return df_predict

def predict_rent(model, X):
    return model.predict(X).item()



# ================== STREAMLIT UI ==================

st.set_page_config(page_title="Rent Price Predictor", layout="centered")

st.title("🏠 Rent Price Prediction")
st.write("Insert the apartment details to estimate the ideal rent price.")

model, features, version = load_model("models/lgbm_regressor_v1.pkl")

st.caption(f"Model trained with pandas {version}")

with st.form("prediction_form"):
    surface = st.number_input("Surface (m²)", min_value=10.0, max_value=500.0, step=5.0)
    city = st.selectbox(
        "City",
        ["Roma", "Milano", "Napoli", "Torino", "Palermo", "Genova",
         "Bologna", "Firenze", "Bari", "Catania", "Verona", "Venezia",
         "Messina", "Padova", "Trieste", "Brescia", "Taranto",
         "Prato", "Parma", "Modena"]
    )
    housing_unit = st.selectbox(
        "Housing type",
            ['Monolocale', 'Bilocale', 'Quadrilocale', 'Trilocale',
            'Appartamento', 'Attico', 'Mansarda', 'Loft', 'Open space',
            'Villa', 'Palazzo', 'Casale', 'Terratetto']
    )
    floor = st.number_input("Floor", min_value=0.0, max_value=6.0, step=1.0)
    num_rooms = st.number_input("Number of rooms", min_value=1.0, max_value=20.0, step=1.0)

    submitted = st.form_submit_button("Predict rent 💰")

# if submitted:
#     X = build_features(surface, city, housing_unit, floor, num_rooms)
#     prediction = int(predict_rent(model, X))

#     st.success(f"💶 **Estimated rent: {prediction} € / month**")

#     with st.expander("Show input features"):
#         st.dataframe(X)

if submitted:
    X = build_features(surface, city, housing_unit, floor, num_rooms)
    prediction = int(predict_rent(model, X))

    # ✅ LOGGING HERE
    import datetime
    import logging
    st.write("Prediction time:", datetime.datetime.now())

    logging.info(f"Prediction: {prediction} | Inputs: {X.to_dict()}")	

    st.success(f"💶 Estimated rent: {prediction} € / month")

    with st.expander("Show input features"):
        st.dataframe(X)



# Write in the cmd Terminal: streamlit run app.py


