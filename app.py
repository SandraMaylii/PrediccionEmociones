import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Lista de emociones sin la clase 'disgust', que ha sido descartada del análisis
CLASSES = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Se carga el modelo previamente entrenado en formato .h5
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# Se carga el clasificador Haar Cascade de OpenCV utilizado para detectar rostros en imágenes
@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

face_cascade = load_face_cascade()

# Se preprocesa la imagen: se convierte a escala de grises y se recorta la región del rostro
def preprocess_face(image: Image.Image):
    img_bgr = np.array(image.convert("RGB"))[:, :, ::-1].copy()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Se realiza la detección de rostros utilizando el clasificador previamente cargado
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # En caso de no detectar ningún rostro, se retorna un mensaje de error
    if len(faces) == 0:
        return None, "No se detectó ningún rostro."

    # Se selecciona el rostro de mayor tamaño (más área), asumiendo que es el principal
    x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    face_crop = gray[y:y+h, x:x+w]

    # Se redimensiona el rostro a 48x48 píxeles y se normaliza para el modelo
    face_resized = cv2.resize(face_crop, (48, 48)) / 255.0
    face_input = face_resized.reshape(1, 48, 48, 1)

    return face_input, None

# Se realiza la predicción emocional y se descarta la clase 'disgust' de los resultados
def predict_emotion(face_input):
    raw_prediction = model.predict(face_input)[0]
    filtered_prediction = np.delete(raw_prediction, 1)  # Se elimina el índice correspondiente a 'disgust'
    return filtered_prediction

# Interfaz principal de la aplicación en Streamlit
st.markdown("""
    <h1 style='text-align: center; color: #1a1a1a;'>Detector de Emociones</h1>
    <h4 style='text-align: center; color: #4f4f4f; font-weight: normal;'>
        Emociones a detectar: <b>angry</b>, <b>fear</b>, <b>happy</b>, <b>neutral</b>, <b>sad</b>, <b>surprise</b>
    </h4>
    <br>
""", unsafe_allow_html=True)

# Se permite al usuario subir una imagen en formato jpg, jpeg o png
uploaded_file = st.file_uploader("Sube una imagen con un rostro", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_container_width=True)

    # Se preprocesa la imagen y se intenta obtener el rostro
    face_input, error_msg = preprocess_face(image)

    # Se muestra un mensaje de error si no se detectó rostro
    if error_msg:
        st.error(error_msg)
    else:
        # Se ejecuta la predicción sobre la imagen preprocesada
        prediction = predict_emotion(face_input)

        # Se valida que la salida del modelo coincida con el número de clases definidas
        if len(prediction) != len(CLASSES):
            st.error("Número de clases del modelo no coincide con CLASSES.")
            st.stop()

        # Se determina la emoción con mayor probabilidad
        predicted_index = np.argmax(prediction)
        predicted_label = CLASSES[predicted_index]
        confidence = prediction[predicted_index] * 100

        # Se muestra el resultado principal de la emoción detectada
        st.success(f"Emoción detectada: **{predicted_label.upper()}** ({confidence:.2f}%)")

        # Se genera un gráfico de barras horizontal para mostrar la distribución de emociones
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.barh(CLASSES, prediction * 100, color='skyblue')
        ax.invert_yaxis()
        for bar, prob in zip(bars, prediction * 100):
            ax.text(prob + 1, bar.get_y() + bar.get_height()/2,
                    f"{prob:.2f}%", va='center', fontsize=9)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Probabilidad (%)")
        ax.set_title("Distribución de emociones")
        st.pyplot(fig)
