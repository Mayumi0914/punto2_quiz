import pandas as pd
import numpy as np
import gensim
import fasttext
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from transformers import BertTokenizer, BertModel
import torch
import stanza
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
stanza.download('en')
nlp_stanza = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma')
nlp_spacy = spacy.load("en_core_web_sm")

stop_nltk = set(stopwords.words('english'))
stop_spacy = nlp_spacy.Defaults.stop_words
stop_todas = list(stop_spacy.union(stop_nltk))

def process_stanza_text(text):
    doc = nlp_stanza(text)
    return [word.lemma for sentence in doc.sentences for word in sentence.words]

def lemmatize_stemming(text):
    ps = PorterStemmer()
    return ps.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    return [lemmatize_stemming(token) for token in gensim.utils.simple_preprocess(text) if token not in stop_todas and len(token) > 3]

def cargar_procesar_datos():
    datos = pd.read_pickle('./training_features_NLP_Encuestas.pkl')
    datos = datos.rename(columns={'corrected': 'essay_text'})
    datos['tokens'] = datos['essay_text'].apply(process_stanza_text)
    return datos[['essay_text', 'essay_id', 'tokens']]

def aplicar_lda(data, use_tfidf=False, num_topics=8):
    documents = data['tokens']
    dictionary = gensim.corpora.Dictionary(documents)
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=500)
    bow_corpus = [dictionary.doc2bow(doc) for doc in documents]

    if use_tfidf:
        tfidf = gensim.models.TfidfModel(bow_corpus)
        corpus = tfidf[bow_corpus]
    else:
        corpus = bow_corpus

    lda_model = gensim.models.LdaMulticore(corpus, num_topics=num_topics, id2word=dictionary, passes=10, workers=2)
    return lda_model, dictionary

def get_bert_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].detach().numpy().flatten()

def aplicar_fasttext():
    ft_model = fasttext.train_unsupervised('./ensayos_NLP_Encuestas.csv', dim=300)
    return ft_model

def generar_embeddings_fasttext(datos, ft_model):
    def average_word_vectors(words, model, num_features=300):
        feature_vector = np.zeros((num_features,), dtype="float64")
        nwords = sum((model.get_word_vector(word) for word in words if word in model.words), start=feature_vector)
        return feature_vector if nwords == 0 else np.divide(feature_vector, nwords)

    embeddings = pd.DataFrame([average_word_vectors(tokens, ft_model) for tokens in datos['tokens']])
    return embeddings

def aplicar_pca_kmeans(embeddings, n_clusters=8):
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings_pca)
    return kmeans

def evaluar_modelos(data, modelos):
    resultados = []

    for modelo in modelos:
        kmeans = aplicar_pca_kmeans(data[modelo + '_embedding'], n_clusters=8)
        data[modelo + '_cluster'] = kmeans.labels_

        for clase in range(8):
            y_true = (data['topic'] == clase).astype(int)
            y_pred = (data[modelo + '_cluster'] == clase).astype(int)

            roc_auc = roc_auc_score(y_true, y_pred)
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)

            resultados.append({'modelo': modelo, 'clase': clase, 'roc_auc': roc_auc, 'accuracy': accuracy, 'f1_score': f1})

    return pd.DataFrame(resultados).sort_values(by='roc_auc', ascending=False)

def pipeline_principal():
    data = cargar_procesar_datos()
    
    lda_model, _ = aplicar_lda(data, use_tfidf=False)
    lda_tfidf_model, _ = aplicar_lda(data, use_tfidf=True)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    data['bert_embedding'] = data['essay_text'].apply(lambda x: get_bert_embedding(x, tokenizer, model))
    
    ft_model = aplicar_fasttext()
    data['fasttext_embedding'] = generar_embeddings_fasttext(data, ft_model)
    
    modelos = ['lda', 'lda_tfidf', 'bert', 'fasttext']
    resultados = evaluar_modelos(data, modelos)
    resultados.to_csv('./resultados_comparacion_modelos.csv', index=False)

if __name__ == "__main__":
    pipeline_principal()
