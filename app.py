import streamlit as st
import joblib
import numpy as np

# Model en features laden
model = joblib.load('model.joblib')
features = joblib.load('features.joblib')

# Titel
st.title('🚲 Bike Sharing — Voorspelling fietsverhuren per uur')
st.write('Voer de kenmerken in om het verwachte totaal aantal fietsverhuren per uur te voorspellen.')

# Invoervelden
yr = st.selectbox('Jaar', options=[0, 1], format_func=lambda x: '2011' if x == 0 else '2012')
mnth = st.slider('Maand', 1, 12, 6)
hr = st.slider('Uur van de dag', 0, 23, 12)
holiday = st.selectbox('Feestdag', options=[0, 1], format_func=lambda x: 'Nee' if x == 0 else 'Ja')
weekday = st.slider('Dag van de week (0=zondag, 6=zaterdag)', 0, 6, 3)
workingday = st.selectbox('Werkdag', options=[0, 1], format_func=lambda x: 'Nee' if x == 0 else 'Ja')
weathersit = st.selectbox('Weersituatie', options=[1, 2, 3, 4], 
                           format_func=lambda x: {1: 'Helder', 2: 'Bewolkt', 3: 'Lichte regen/sneeuw', 4: 'Zwaar weer'}[x])
temp = st.slider('Temperatuur (genormaliseerd)', 0.0, 1.0, 0.5)
hum = st.slider('Luchtvochtigheid (genormaliseerd)', 0.0, 1.0, 0.5)
windspeed = st.slider('Windsnelheid (genormaliseerd)', 0.0, 1.0, 0.2)

# Voorspelling
if st.button('Voorspel'):
    input_data = np.array([[yr, mnth, hr, holiday, weekday, workingday, weathersit, temp, hum, windspeed]])
    prediction = model.predict(input_data)[0]
    st.success(f'🚲 Verwacht totaal aantal fietsverhuren: **{prediction:.0f}** per uur')
