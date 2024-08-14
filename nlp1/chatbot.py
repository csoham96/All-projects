import speech_recognition as sr
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sentence_transformers import SentenceTransformer
import faiss
import openai

class Chatbot:
    def __init__(self,audio_file):
        # Initialize components
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = faiss.IndexFlatL2(embedding_dim=786)  # Replace embedding_dim with actual value
        self.id_mapping = {}
        self.audio_file=audio_file

    def speech_to_text(self, audio_file):
        r = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio= r.listen(source)
        try:
            text = r.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            print("Could not understand audio")

        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))

    def embed_text(self, text):
        embeddings = self.model.encode(text)
        return embeddings

    def add_to_index(self, embeddings, ids):
        self.index.add(embeddings)
        for i, embed in enumerate(embeddings):
            self.id_mapping[embed.tobytes()] = ids[i]

    def search_index(self, query_embedding, k):
        text=self.speech_to_text(self.audio_file)
        query_embedding=self.embed_text(text)
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        return distances.tolist()[0], self.get_data_by_indices(indices.tolist()[0])

    def get_data_by_indices(self, indices):
        data = []
        for idx in indices:
            data.append(get_data_from_id(self.id_mapping[self.index.ids[0][idx]]))
        return data
    def generate_response(self, prompt, context):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt + "\n" + context,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )
        return response.choices[0].text.strip()

    def chatbot(self, query):
            # Preprocess query
            query_embedding = self.embed_text(query)
            # Search vector database
            relevant_indices = self.search_index(index, query_embedding, k)
            # Retrieve relevant speech data
            relevant_text = self.get_text_from_indices(relevant_indices)
            # Generate response using LLM
            response = self.generate_response(query, relevant_text)
            return response
    

