from flask import Flask, request, jsonify
from rag_chain import create_rag_chain

# Crear la app
app = Flask(__name__)

# Cargar la cadena RAG (esto toma unos segundos al inicio)
qa_chain = create_rag_chain()

# Ruta principal para hacer preguntas
@app.route("/chat", methods=["POST"])
def chat():
    # Obtener la pregunta del usuario
    user_input = request.json.get("question")

    if not user_input:
        return jsonify({"error": "Debes enviar una pregunta en el campo 'question'"}), 400

    # Ejecutar la cadena de pregunta-respuesta
    raw_answer = qa_chain.run(user_input)

    # Extraer solo la parte después de la última ocurrencia de "Answer:"
    if "Answer:" in raw_answer:
        answer = raw_answer.split("Answer:")[-1].strip()
    else:
        answer = raw_answer.strip()

    return jsonify({"answer": answer})

# Iniciar servidor Flask
if __name__ == "__main__":
    app.run(debug=True)