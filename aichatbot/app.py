import fitz  # PyMuPDF
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import string
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')

def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

def preprocess_text(text):
    sentences = sent_tokenize(text)
    tokens = [word_tokenize(sentence) for sentence in sentences]
    tokens = [[token.lower() for token in sentence] for sentence in tokens]
    tokens = [[token for token in sentence if token not in string.punctuation] for sentence in tokens]
    tokens = [[token for token in sentence if token not in stopwords.words('english')] for sentence in tokens]
    return [" ".join(sentence) for sentence in tokens], sentences

pdf_path = 'aichatbot\\report.pdf'  
pdf_text = extract_text_from_pdf(pdf_path)
processed_text, original_sentences = preprocess_text(pdf_text)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_text)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    response = retrieve_information(user_input)
    return jsonify({"response": response})

def retrieve_information(query):
    query_vec = vectorizer.transform([query.lower()])
    results = (X * query_vec.T).toarray()
    relevant_indices = np.argsort(results.flatten())[::-1]
    top_n = 3  # Number of top results to return
    relevant_sentences = [original_sentences[i] for i in relevant_indices[:top_n]]
    return " ".join(relevant_sentences)

if __name__ == "__main__":
    app.run(debug=True, port=5001,host='0.0.0.0')

# import fitz  # PyMuPDF
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize, sent_tokenize
# import string
# from flask import Flask, request, jsonify, render_template
# import numpy as np
# import torch
# from transformers import BertTokenizer, BertModel

# nltk.download('punkt')
# nltk.download('stopwords')

# def extract_text_from_pdf(pdf_path):
#     document = fitz.open(pdf_path)
#     text = ""
#     for page_num in range(len(document)):
#         page = document.load_page(page_num)
#         text += page.get_text()
#     return text

# def preprocess_text(text):
#     sentences = sent_tokenize(text)
#     tokens = [word_tokenize(sentence) for sentence in sentences]
#     tokens = [[token.lower() for token in sentence] for sentence in tokens]
#     tokens = [[token for token in sentence if token not in string.punctuation] for sentence in tokens]
#     tokens = [[token for token in sentence if token not in stopwords.words('english')] for sentence in tokens]
#     return [" ".join(sentence) for sentence in tokens], sentences

# def get_sentence_embeddings(sentences, model, tokenizer):
#     embeddings = []
#     for sentence in sentences:
#         inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=512)
#         outputs = model(**inputs)
#         embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy())
#     return np.array(embeddings)

# def cosine_similarity(a, b):
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# pdf_path = 'C:\\Users\\User\\Documents\\Python\\flask\\image\\report.pdf'  
# pdf_text = extract_text_from_pdf(pdf_path)
# processed_text, original_sentences = preprocess_text(pdf_text)

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')

# sentence_embeddings = get_sentence_embeddings(processed_text, model, tokenizer)

# app = Flask(__name__)

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/chat", methods=["POST"])
# def chat():
#     user_input = request.json.get("message")
#     response = retrieve_information(user_input)
#     return jsonify({"response": response})

# def retrieve_information(query):
#     query_embedding = get_sentence_embeddings([query], model, tokenizer)[0]
#     similarities = [cosine_similarity(query_embedding, sentence_embedding) for sentence_embedding in sentence_embeddings]
#     relevant_indices = np.argsort(similarities)[::-1]
#     top_n = 3  # Number of top results to return

#     relevant_sentences = []
#     for i in relevant_indices[:top_n]:
#         if i > 0:
#             relevant_sentences.append(original_sentences[i - 1])
#         relevant_sentences.append(original_sentences[i])
#         if i < len(original_sentences) - 1:
#             relevant_sentences.append(original_sentences[i + 1])
    
#     return " ".join(relevant_sentences)

# if __name__ == "__main__":
#     app.run(debug=True)
