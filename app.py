import streamlit as st
import joblib
import numpy as np
import sqlite3
import pandas as pd
import os
from datetime import datetime, timedelta

# --- Database functies ---
def init_db():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            yr INTEGER,
            mnth INTEGER,
            hr INTEGER,
            holiday INTEGER,
            weekday INTEGER,
            workingday INTEGER,
            weathersit INTEGER,
            temp REAL,
            hum REAL,
            windspeed REAL,
            prediction REAL,
            source TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_prediction(input_values, prediction, source='manual'):
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO predictions (timestamp, yr, mnth, hr, holiday, weekday, 
                                  workingday, weathersit, temp, hum, windspeed, prediction, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), *input_values, prediction, source))
    conn.commit()
    conn.close()

def get_predictions(source=None):
    conn = sqlite3.connect('predictions.db')
    if source:
        df = pd.read_sql_query('SELECT * FROM predictions WHERE source = ? ORDER BY timestamp DESC', conn, params=[source])
    else:
        df = pd.read_sql_query('SELECT * FROM predictions ORDER BY timestamp DESC', conn)
    conn.close()
    return df

# --- Database initialiseren ---
init_db()

# --- Model laden ---
if os.path.exists('modellen/model.joblib'):
    model = joblib.load('modellen/model.joblib')
    features = joblib.load('modellen/features.joblib')
else:
    model = joblib.load('model.joblib')
    features = joblib.load('features.joblib')

# --- App interface ---
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
    input_data = pd.DataFrame([[yr, mnth, hr, holiday, weekday, workingday, weathersit, temp, hum, windspeed]], 
                               columns=features)
    prediction = model.predict(input_data)[0]
    
    input_values = [yr, mnth, hr, holiday, weekday, workingday, weathersit, temp, hum, windspeed]
    save_prediction(input_values, prediction, source='manual')
    
    st.success(f'🚲 Verwacht totaal aantal fietsverhuren: **{prediction:.0f}** per uur')

# --- Simuleer voorspellingen met synthetische data ---
st.markdown('---')
st.subheader('🔄 Simuleer voorspellingen met synthetische data')

col1, col2 = st.columns(2)
with col1:
    sim_start = st.date_input('Startdatum simulatie', value=datetime.now().date() - timedelta(days=30))
with col2:
    sim_end = st.date_input('Einddatum simulatie', value=datetime.now().date() + timedelta(days=30))

if st.button('Simuleer voorspellingen'):
    if sim_start >= sim_end:
        st.error('Startdatum moet vóór einddatum liggen.')
    else:
        if os.path.exists('data/synthetic_data.csv'):
            synthetic_path = 'data/synthetic_data.csv'
        elif os.path.exists('synthetic_data.csv'):
            synthetic_path = 'synthetic_data.csv'
        else:
            synthetic_path = None
        
        if synthetic_path:
            synthetic = pd.read_csv(synthetic_path)
            predictions = model.predict(synthetic[features])
            
            # Gespreide timestamps over de gekozen periode
            timestamps = pd.date_range(start=sim_start, end=sim_end, periods=len(synthetic))
            
            conn = sqlite3.connect('predictions.db')
            c = conn.cursor()
            for i, row in synthetic.iterrows():
                c.execute('''
                    INSERT INTO predictions (timestamp, yr, mnth, hr, holiday, weekday, 
                                              workingday, weathersit, temp, hum, windspeed, prediction, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (timestamps[i].strftime('%Y-%m-%d %H:%M:%S'), 
                      int(row['yr']), int(row['mnth']), int(row['hr']), int(row['holiday']),
                      int(row['weekday']), int(row['workingday']), int(row['weathersit']),
                      row['temp'], row['hum'], row['windspeed'], predictions[i], 'synthetic'))
            conn.commit()
            conn.close()
            
            st.success(f'✅ {len(synthetic)} synthetische voorspellingen opgeslagen van {sim_start} tot {sim_end}!')
        else:
            st.error('Synthetische dataset niet gevonden.')

# --- Prediction History ---
st.markdown('---')
st.subheader('📊 Prediction History')

history = get_predictions()

if len(history) > 0:
    st.write(f'Totaal aantal opgeslagen voorspellingen: **{len(history)}**')
    
    # Datumfilter voor weergave
    history['timestamp'] = pd.to_datetime(history['timestamp'])
    
    st.write('**Filter op periode:**')
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        view_start = st.date_input('Van', value=history['timestamp'].min().date(), key='view_start')
    with filter_col2:
        view_end = st.date_input('Tot', value=history['timestamp'].max().date(), key='view_end')
    
    # Filteren
    filtered = history[(history['timestamp'].dt.date >= view_start) & (history['timestamp'].dt.date <= view_end)]
    
    st.write(f'Voorspellingen in geselecteerde periode: **{len(filtered)}**')
    st.dataframe(filtered, use_container_width=True)
    
    if len(filtered) > 0:
        # Grafiek met onderscheid tussen handmatig en synthetisch
        st.subheader('📈 Voorspellingen over tijd')
        filtered = filtered.sort_values('timestamp')
        
        manual_data = filtered[filtered['source'] == 'manual'][['timestamp', 'prediction']].rename(columns={'prediction': 'Handmatig'}).set_index('timestamp')
        synth_data = filtered[filtered['source'] == 'synthetic'][['timestamp', 'prediction']].rename(columns={'prediction': 'Synthetisch'}).set_index('timestamp')
        
        chart_df = pd.concat([manual_data, synth_data], axis=1)
        st.line_chart(chart_df, color=['#FF4B4B', '#4B8BFF'])
else:
    st.info('Nog geen voorspellingen opgeslagen. Maak een voorspelling hierboven.')
