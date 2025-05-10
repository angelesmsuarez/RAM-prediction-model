import streamlit as st
import pandas as pd
import joblib
import re
import requests
import io

url = "https://drive.google.com/uc?export=download&id=1A2B3C4D5E6F7G8H9I"  # <-- usa tu ID
response = requests.get(url)
df = pd.read_parquet(io.BytesIO(response.content))

# 🔄 Cargar modelo y encoders
model = joblib.load('modelo_ram_xgboost_afinado.pkl')
le_sex = joblib.load('le_sex.pkl')
le_medic = joblib.load('le_medic.pkl')

# 📂 Leer dataset para coincidencias
df_raw = pd.read_csv('C:/Users/encin/Angeles/faers_data_limpio.csv', low_memory=False)

# ✅ Filtrar medicamentos con al menos 20 reportes
med_counts = df_raw['medicinalproduct'].value_counts()
medicamentos_validos = med_counts[med_counts >= 20].index.tolist()

# 🔎 Reacciones que NO son RAMs reales
excluir_reacciones = [
    'off label use',
    'drug ineffective',
    'product use in unapproved indication',
    'inappropriate schedule of product administration'
    'Maternal exposure during pregnancy'
    'Inappropriate schedule of product administration'
    
]

st.title('💊 Sistema de Alerta de RAMs')

# 👉 Inputs del usuario
age = st.number_input('Edad del paciente', min_value=0, max_value=120, value=30)
sex = st.selectbox('Sexo del paciente', le_sex.classes_)
medicamento = st.selectbox('Medicamento', medicamentos_validos)

if st.button('🔍 Consultar riesgo y coincidencias'):
    # Codificar entrada
    sex_enc = le_sex.transform([sex])[0]
    try:
        medic_enc = le_medic.transform([medicamento])[0]
    except ValueError:
        st.error("⚠️ El medicamento seleccionado no está en el encoder. Intenta con otro.")
        st.stop()

    # 🔄 DataFrame de entrada para predicción
    input_df = pd.DataFrame({
        'patientonsetage': [age],
        'patientsex': [sex_enc],
        'medicinalproduct': [medic_enc]
    })

    # 🧠 Predicción
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df).max()

    # Mostrar resultados
    st.subheader('📢 Resultados:')
    st.write(f"⚠️ **¿Evento grave?:** {'Sí' if pred == 1 else 'No'}")
    st.write(f"📊 **Probabilidad del modelo:** {prob:.2%}")

    # ✍️ Breve explicación de la probabilidad
    st.caption(
        "ℹ️ La probabilidad indica cuán seguro está el modelo de su predicción actual. "
        "Valores cercanos a 100% reflejan alta confianza; valores bajos (<50%) sugieren baja certeza."
    )

    st.markdown("---")

    # 🔎 Buscar coincidencias
    st.subheader('🗂️ Reacciones reportadas en casos similares:')

    # Filtrar coincidencias (edad exacta y sexo)
    df_similares = df_raw[
        (df_raw['patientonsetage'].round() == age) &
        (df_raw['patientsex'] == sex)
    ]

    # Buscar medicamentos similares (match por nombre base)
    pattern = re.escape(medicamento.split()[0])
    df_similares = df_similares[
        df_similares['medicinalproduct'].str.contains(pattern, case=False, na=False)
    ]

    # Excluir "RAMs" irrelevantes
    df_similares = df_similares[
        ~df_similares['reactionmeddrapt'].str.lower().isin(excluir_reacciones)
    ]

    # Mostrar hasta 5 reacciones (sin contar número de reportes)
    if not df_similares.empty:
        top_reacciones = (
            df_similares['reactionmeddrapt']
            .value_counts()
            .head(5)
            .index.tolist()
        )
        for i, reaccion in enumerate(top_reacciones, start=1):
            st.write(f"**{i}. {reaccion}**")
    else:
        st.info('No se encontraron coincidencias para las características seleccionadas.')

# Footer
st.caption('Desarrollado para tesis - Sistema de predicción de RAMs.')
