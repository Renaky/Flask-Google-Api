from flask import Flask, jsonify, request
import numpy as np
import google.generativeai as genai
import pickle
from flask_cors import CORS
from dotenv import load_dotenv
import os

load_dotenv()
app = Flask(__name__)
CORS(app)  # Initialize CORS for the entire application

model = 'models/text-embedding-004'
modeloEmbeddings = pickle.load(open('datasetEmbeddings.pkl','rb'))
chave_secreta = os.getenv('API_KEY')
genai.configure(api_key=chave_secreta)

def gerarBuscarConsulta(consulta, dataset):
    embedding_consulta = genai.embed_content(model=model,
                                             content=consulta,
                                             task_type="retrieval_query")
    produtos_escalares = np.dot(np.stack(dataset["Embeddings"]), embedding_consulta['embedding']) # Cálculo da similaridade
    indice = np.argmax(produtos_escalares)
    
    # Captura as informações desejadas da linha correspondente
    cidade = dataset.iloc[indice]['Cidade']
    estado = dataset.iloc[indice]['Estado']
    principais_atracoes = dataset.iloc[indice]['Principais Atrações']
    melhores_restaurantes = dataset.iloc[indice]['Melhores Restaurantes']
    atividades_sugeridas = dataset.iloc[indice]['Atividades Sugeridas']
    aeroporto_principal = dataset.iloc[indice]['Aeroporto Principal']
    
    # Formata a resposta com todas as informações
    resposta = (f"Cidade: {cidade}\n"
                f"Estado: {estado}\n"
                f"Principais Atrações: {principais_atracoes}\n"
                f"Melhores Restaurantes: {melhores_restaurantes}\n"
                f"Atividades Sugeridas: {atividades_sugeridas}\n"
                f"Aeroporto Principal: {aeroporto_principal}")
    
    return resposta

model2 = genai.GenerativeModel( model_name="gemini-1.0-pro")

@app.route("/")
def home():
    try:
        consulta = "O que você pode fazer?"
        resultado = gerarBuscarConsulta(consulta, modeloEmbeddings)

 # Criar um prompt personalizado para gerar a resposta
        prompt = (f"A seguir está uma informação retirada do banco de dados sobre um bom lugar para ver cachoeiras:\n\n"
                    f"{resultado}\n\n"
                    f"Agora, por favor, crie uma resposta personalizada e amigável para um usuário que perguntou: '{consulta}'.")
        
        # Gerar o conteúdo
        response = model2.generate_content(prompt)
        
        # Verificar se a resposta contém 'text'
        if hasattr(response, 'text'):
            return response.text
        else:
            return "A resposta não contém o campo 'text'.", 500
    except Exception as e:
        return f"Ocorreu um erro ao processar a solicitação: {e}", 500

@app.route("/api", methods=["POST"])
def results():
    # Verifique a chave de autorização
    auth_key = request.headers.get("Authorization")
    if auth_key != chave_secreta:
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        data = request.get_json(force=True)
        consulta = data["consulta"]
        resultado = gerarBuscarConsulta(consulta, modeloEmbeddings)
        prompt = (f"A seguir está uma informação retirada do banco de dados sobre um bom lugar para ver cachoeiras:\n\n"
                    f"{resultado}\n\n"
                    f"Agora, por favor, crie uma resposta personalizada e amigável para um usuário que perguntou: '{consulta}'.")
        
        # Gerar o conteúdo
        response = model2.generate_content(prompt)
        
        # Verificar se a resposta contém 'text'
        if hasattr(response, 'text'):
            return jsonify({"mensagem": response.text})
        else:
            return jsonify({"error": "A resposta não contém o campo 'text'."}), 500
    except Exception as e:
        return jsonify({"error": f"Ocorreu um erro ao processar a solicitação: {e}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
