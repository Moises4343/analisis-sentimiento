from flask import Flask, request, jsonify, make_response
from flask_restful import Resource, Api
from textblob import TextBlob
from googletrans import Translator
import re
import nltk
import logging
import os
from nltk.corpus import stopwords
from transformers import BertTokenizer


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

nltk.download('stopwords')
stop_words = set(stopwords.words('spanish'))

app = Flask(__name__)
api = Api(app)

translator = Translator()

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')


groserias_path = 'tokenized_groserias.txt'
if not os.path.exists(groserias_path):
    logging.error(f"{groserias_path} no se encuentra en el directorio actual.")
    raise FileNotFoundError(f"{groserias_path} no se encuentra en el directorio actual.")
with open(groserias_path, 'r', encoding='iso-8859-1') as f:
    tokenized_groserias = set(f.read().splitlines())

class SentimentAnalysis(Resource):
    def preprocess_text(self, text):
        try:
            text = text.replace("4", "a").replace("3", "e").replace("1", "i").replace("0", "o")
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\d+', '', text)
            text = text.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u').replace('ñ', 'n')
            text = ' '.join([word for word in text.split() if word not in stop_words])
            text = text.strip()
            return text
        except Exception as e:
            logging.error(f"Error en el preprocesamiento del texto: {e}")
            raise

    def detect_groserias(self, text):
        try:
            tokens = bert_tokenizer.tokenize(text)
            for token in tokens:
                if token in tokenized_groserias:
                    return True
            return False
        except Exception as e:
            logging.error(f"Error en la detección de groserías: {e}")
            raise

    def post(self):
        try:
            data = request.get_json()
            logging.info(f"Datos recibidos: {data}")
            text = data.get('text', '')

            if not text:
                logging.error("No se proporcionó texto para el análisis.")
                return make_response(jsonify({'error': 'No se proporcionó texto para el análisis.'}), 400)
            
            cleaned_text = self.preprocess_text(text)

            if self.detect_groserias(cleaned_text):
                logging.error("El texto contiene groserías y no puede ser procesado.")
                return make_response(jsonify({'error': 'El texto contiene groserías y no puede ser procesado.'}), 400)

            translated_text = translator.translate(cleaned_text, src='es', dest='en').text
            blob = TextBlob(translated_text)
            sentiment = blob.sentiment
            
            response = {
                'original_text': text,
                'cleaned_text': cleaned_text,
                'translated_text': translated_text,
                'sentiment': {
                    'polarity': sentiment.polarity,
                    'subjectivity': sentiment.subjectivity
                }
            }
            logging.info(f"Respuesta generada: {response}")
            return make_response(jsonify(response), 200)
        except Exception as e:
            logging.error(f"Error en la solicitud POST: {e}")
            return make_response(jsonify({'error': 'Error interno del servidor'}), 500)

api.add_resource(SentimentAnalysis, '/sentiment')

@app.route('/')
def home():
    return 'Bienvenido a la API de Análisis de Sentimientos'

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
