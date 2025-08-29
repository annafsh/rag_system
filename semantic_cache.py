from functools import cache
import faiss            # Efficient similarity search over vector embeddings
import json             # Read/write cache from a JSON file
import numpy as np      # Numerical operations on embeddings
# from sentence_transformers import SentenceTransformer  # Load Nomic embed model
import time             # Measure latency

# OS operations
import os

# SSL configuration
import ssl
import urllib3

from nomic import embed
import numpy as np

# Disable SSL verification and warnings for development
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['CURL_CA_BUNDLE'] = ''
# Additional SSL bypass for requests library used by nomic
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_VERIFY'] = 'false'
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Monkey patch requests to disable SSL for nomic library
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class NoSSLAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        kwargs['ssl_context'] = ssl._create_unverified_context()
        return super().init_poolmanager(*args, **kwargs)

# Apply SSL bypass globally
session = requests.Session()
session.mount('https://', NoSSLAdapter())
requests.Session = lambda: session

# Environment variable loading
from dotenv import load_dotenv  # Load environment variables from .env file

# Load environment variables
load_dotenv()

class SemanticCaching:

    def __init__(self, json_file='cache.json', clear_on_init=True):
        # Initialize Faiss index with Euclidean distance
        self.index = faiss.IndexFlatL2(768)  # Use IndexFlatL2 with Euclidean distance
        if self.index.is_trained:
            print('Index trained')

        # Initialize Nomic embedding (replacing SentenceTransformer)
        # self.encoder = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)


        # Uncomment the following lines to use DialoGPT for question generation
        # self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
        # self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

        # Euclidean distance threshold for cache hits (lower = more similar)
        self.euclidean_threshold = 0.2

        # JSON file to persist cache entries
        self.json_file = json_file

        # Load cache or clear already loaded cache
        if clear_on_init:
            self.clear_cache()
        else:
            self.load_cache()

    def save_cache(self):
        """Persist cache back to disk."""
        with open(self.json_file, 'w') as file:
            json.dump(self.cache, file)        

    def clear_cache(self):
        """
        Clears in-memory cache, resets FAISS index, and overwrites cache.json with an empty structure.
        """
        self.cache = {
            'questions': [],
            'embeddings': [],
            'answers': []
        }
        self.index = faiss.IndexFlatL2(768)  # Reinitialize FAISS index
        self.save_cache()
        print("Semantic cache cleared.")

    def load_cache(self):
        """Load existing cache and rebuild FAISS index."""
        try:
            with open(self.json_file, 'r') as file:
                self.cache = json.load(file)
                
            # Remove legacy response_text field if it exists
            if 'response_text' in self.cache:
                del self.cache['response_text']
                
            # Validate cache structure
            if not all(key in self.cache for key in ['questions', 'embeddings', 'answers']):
                raise ValueError("Invalid cache structure")
                
            # Ensure all lists have the same length
            lengths = [len(self.cache[key]) for key in ['questions', 'embeddings', 'answers']]
            if len(set(lengths)) > 1:
                raise ValueError("Cache lists have inconsistent lengths")
                
            # Rebuild FAISS index from cached embeddings
            if self.cache['embeddings']:
                embeddings_array = np.array(self.cache['embeddings']).astype('float32')
                self.index.add(embeddings_array)
                print(f"Loaded {len(self.cache['questions'])} cached entries and rebuilt FAISS index")
            else:
                print("No cached entries found")
                
        except FileNotFoundError:
            # Structure: lists of questions, embeddings, and answers
            self.cache = {'questions': [], 'embeddings': [], 'answers': []}
            print("No existing cache found, starting with empty cache")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Cache corruption detected: {e}. Starting with empty cache.")
            self.cache = {'questions': [], 'embeddings': [], 'answers': []}

    def get_text_embeddings(self, text: str):
        """
        Converts input text into a dense embedding using nomic local deployment
        """
        try:
            print("Attempting local Nomic embedding...")
            
            # Additional SSL bypass for requests library used by nomic
            #sometimes needs to be set directly with the path here
            #os.environ['REQUESTS_CA_BUNDLE'] = ""
            
            os.environ['SSL_VERIFY'] = 'false'
            old_context = ssl._create_default_https_context
            ssl._create_default_https_context = ssl._create_unverified_context
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            
            try:
                output = embed.text(
                    texts=[text],
                    model='nomic-embed-text-v1.5',
                    task_type="search_query",
                    inference_mode='local',
                    dimensionality=768
                )
                print("✓ Successfully used local Nomic embedding")
                return output['embeddings'][0]  # Return the embedding for the single text
            finally:
                # Restore original SSL context
                ssl._create_default_https_context = old_context
                
        except Exception as e:
            print(f"Local Nomic embedding failed: {e}")
            raise RuntimeError(f"Error generating embeddings: {e}")

    def ask(self, question: str) -> str:
        """
        Returns a cached answer if within threshold, otherwise generates,
        caches, and returns a new answer.
        """
        start_time = time.time()
        try:
            # Encode the incoming question using nomic
            embedding = self.get_text_embeddings(question)
            embedding = np.array([embedding])  # Convert to numpy array with proper shape

            # Search for the nearest neighbor in the index
            D, I = self.index.search(embedding, 1)

            answer = None
            processing_info = None

            # 3) If a neighbor exists and is within threshold → cache hit
            if D[0] >= 0:
                if I[0][0] != -1 and D[0][0] <= self.euclidean_threshold:
                    row_id = int(I[0][0])
                    print(f'Cache hit at row: {row_id} with score {1 - D[0][0]}') #score inversed to show similarity
                    print(f"Time taken: {time.time() - start_time:.3f}s")
                    answer = self.cache['answers'][row_id]
                    processing_info = f"Cache hit at row: {row_id} with score {1 - D[0][0]}"

            return question, embedding, answer, processing_info

        except Exception as e:
            raise RuntimeError(f"Error during 'ask' method: {e}")

    def add_to_cache(self, question: str, embedding: np.ndarray, answer: str):
        """Add new entry to cache and update FAISS index."""
        try:
            # Ensure embedding is the correct shape and type
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
            embedding = embedding.astype('float32')
            
            # Append new entry to cache
            self.cache['questions'].append(question)
            self.cache['embeddings'].append(embedding[0].tolist())
            self.cache['answers'].append(answer)
            
            # Add to FAISS index
            self.index.add(embedding)
            
            # Persist to disk
            self.save_cache()
            
            print(f"Cache updated for question: {question}")
            
        except Exception as e:
            print(f"Error adding to cache: {e}")
            raise

