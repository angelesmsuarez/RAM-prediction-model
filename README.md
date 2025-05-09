Sistema de Alerta de RAMs (Reacciones Adversas a Medicamentos)
Este proyecto implementa un modelo de Machine Learning basado en XGBoost para predecir la probabilidad y la gravedad de reacciones adversas a medicamentos (RAMs) en función de la edad, el sexo y el medicamento administrado. La aplicación está desarrollada con Streamlit e incluye una interfaz interactiva para la consulta de resultados.
¿Cómo ejecutar la aplicación?

1. Descargar los datos
Archivo necesario:
faers_data_ml_ready.parquet
El archivo puede descargarse desde el siguiente enlace público:
https://drive.google.com/file/d/1VV5zm5qSMJ8tLycIk7j-ckXJUFr5F8K1/view?usp=sharing
Nota: El archivo debe colocarse en la raíz del proyecto (en la misma ubicación que app.py).

2. Instalar dependencias
bash
Copy
pip install -r requirements.txt

3. Ejecutar la aplicación
bash
Copy
streamlit run app.py
Tecnologías y librerías utilizadas
Python 3.12
Streamlit
XGBoost
Pandas
Scikit-learn

Notas adicionales
Asegúrese de que el archivo .parquet esté disponible en la carpeta principal del proyecto antes de ejecutar la aplicación.
Si necesita regenerar el archivo desde un archivo .csv, puede utilizar el siguiente código:
python
Copy
import pandas as pd
df = pd.read_csv('ruta_al_archivo.csv')
df.to_parquet('faers_data_ml_ready.parquet')

Si existen dudas, comunicarse con la autora del proyecto, Angeles Muñoz.
angy9902@outlook.com
