from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle


app = Flask(__name__)
CORS(app)  # Esto habilita CORS para todas las rutas

with open("modelo_sentimiento.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

@app.route("/analizar", methods=["POST"])
def analizar():
    data = request.get_json()
    texto = data.get("texto", "")
    if not texto:
        return jsonify({"error": "No se recibió texto"}), 400

    texto_vec = vectorizer.transform([texto])
    pred = model.predict(texto_vec)[0]
    sentimiento = "Positiva" if pred == 1 else "Negativa"
    return jsonify({"sentimiento": sentimiento})

@app.route("/ping")
def ping():
    return "API activa"

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Render define el puerto
    app.run(host="0.0.0.0", port=port)