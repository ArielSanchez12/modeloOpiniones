# ===== IMPORTACIONES NECESARIAS =====
from flask import Flask, request, jsonify  # Flask para crear la API
from flask_cors import CORS  # Para permitir peticiones desde otro dominio (ej. Vercel)
import pickle  # Para cargar el modelo previamente entrenado

# ===== CONFIGURACIÓN DE LA APLICACIÓN FLASK =====
app = Flask(__name__)  # Se crea una instancia de la app Flask
CORS(app)  # Esto habilita CORS (Cross-Origin Resource Sharing) para permitir peticiones desde cualquier origen

# ===== CARGA DEL MODELO ENTRENADO =====
# Se carga el vectorizador TF-IDF y el modelo de regresión logística desde el archivo .pkl
with open("modelo_sentimiento.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

# ===== RUTA PRINCIPAL PARA ANALIZAR OPINIONES =====
@app.route("/analizar", methods=["POST"])
def analizar():
    data = request.get_json()  # Se obtiene el JSON enviado en la petición
    texto = data.get("texto", "")  # Se extrae el campo "texto" del JSON

    # Validación: Si no se recibe texto, se devuelve error 400
    if not texto:
        return jsonify({"error": "No se recibió texto"}), 400

    # Transformar el texto recibido con el vectorizador entrenado
    texto_vec = vectorizer.transform([texto])
    
    # Usar el modelo para predecir si es positivo (1) o negativo (0)
    pred = model.predict(texto_vec)[0]
    sentimiento = "Positiva" if pred == 1 else "Negativa"

    # Devolver el resultado en formato JSON
    return jsonify({"sentimiento": sentimiento})

# ===== RUTA DE PRUEBA PARA VER SI LA API ESTÁ ACTIVA =====
@app.route("/ping")
def ping():
    return "API activa"

# ===== EJECUTAR LA APLICACIÓN LOCALMENTE =====
# Ejecuta la API en el puerto 5000.
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)