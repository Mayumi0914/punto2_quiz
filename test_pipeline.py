import unittest
import pandas as pd
from transformers import BertTokenizer, BertModel
from p2 import cargar_procesar_datos, aplicar_lda, get_bert_embedding, aplicar_fasttext, generar_embeddings_fasttext

class TestPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Cambia esta ruta a la ubicación real de tu archivo de datos
        cls.data_path = './training_features_NLP_Encuestas.pkl'
        cls.data = cargar_procesar_datos()
        cls.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        cls.model = BertModel.from_pretrained('bert-base-uncased')
        cls.ft_model = aplicar_fasttext()
    
    def test_cargar_procesar_datos(self):
        self.assertFalse(self.data.empty, "El DataFrame cargado está vacío. Verifica la ruta del archivo de datos.")
        self.assertIn('tokens', self.data.columns, "La columna 'tokens' no está presente en los datos.")
    
    def test_aplicar_lda(self):
        lda_model, _ = aplicar_lda(self.data, use_tfidf=False)
        self.assertIsNotNone(lda_model, "El modelo LDA no se ha generado correctamente.")
    
    def test_get_bert_embedding(self):
        if not self.data.empty:
            embedding = get_bert_embedding(self.data['essay_text'].iloc[0], self.tokenizer, self.model)
            self.assertEqual(embedding.shape[0], 768, "El embedding BERT no tiene la dimensión esperada.")
    
    def test_aplicar_fasttext(self):
        self.assertIsNotNone(self.ft_model, "El modelo FastText no se ha generado correctamente.")
    
    def test_generar_embeddings_fasttext(self):
        embeddings = generar_embeddings_fasttext(self.data, self.ft_model)
        self.assertFalse(embeddings.empty, "Los embeddings generados con FastText están vacíos.")
        self.assertEqual(embeddings.shape[1], 300, "La dimensión de los embeddings FastText no es correcta.")

if __name__ == '__main__':
    unittest.main()
