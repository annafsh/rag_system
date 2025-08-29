import requests             # Used for making HTTP requests (e.g., calling ARES API for live internet queries)
import json                 # For parsing and structuring JSON data (especially OpenAI and routing responses)

# OS operations
import os                   # Useful for accessing environment variables and managing paths

# Environment variable loading
from dotenv import load_dotenv  # Load environment variables from .env file

# OpenAI API client
from openai import AzureOpenAI   # Azure OpenAI client library to interface with GPT models for routing and generation
from openai import OpenAIError  # For handling OpenAI API errors

# Text processing
import re                   # Regular expressions for cleaning or preprocessing inputs (if needed)

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

# Vector database client
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv()


class TenkAgent:
    """
    A class to handle 10-K document queries using RAG (Retrieval-Augmented Generation).
    
    This class encapsulates the functionality for embedding generation, document retrieval
    from Qdrant vector database, and response generation using Azure OpenAI.
    """

    def __init__(self, openaiclient, qdrant_path: str = "..\\qdrant_data"):
        """
        Initialize the TenkAgent with Qdrant client and Azure OpenAI client.
        
        Args:
            qdrant_path (str): Path to the Qdrant database
        """
        # Initialize Qdrant client
        try:
            self.client = QdrantClient(path=qdrant_path)
            print("Successfully initialized Qdrant client")
        except Exception as e:
            print(f"Warning: Could not initialize Qdrant client: {e}")
            self.client = None

        self.openaiclient = openaiclient
        
        # Define mapping of routing labels to their respective Qdrant collections
        self.collections = {
            "OPENAI_QUERY": "opnai_data",           # Collection of OpenAI documentation embeddings
            "10K_DOCUMENT_QUERY": "10k_data"        # Collection of 10-K financial document embeddings
        }

    def get_text_embeddings(self, text: str):
        """
        Converts input text into a dense embedding using nomic
        These embeddings are used to query Qdrant for semantically relevant document chunks.

        Args:
            text (str): The input text or query from the user.

        Returns:
            list: A fixed-size vector representing the semantic meaning of the input.
        """
        
        # Try local Nomic embedding first
        try:
            print("Attempting local Nomic embedding...")
            
            # Additional SSL bypass attempt for nomic
            import ssl
            

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

    def retrieve_and_response(self, user_query: str, action: str):
        """
        Retrieves relevant text chunks from the appropriate Qdrant collection
        based on the query type, then generates a response using RAG.

        This function powers the retrieval and response generation pipeline
        for queries that are classified as either OPENAI-related or 10-K related.
        It uses semantic search to fetch relevant context from a Qdrant vector store
        and then generates a response using that context via a RAG prompt.

        Args:
            user_query (str): The user's input question.
            action (str): The classification label from the router (e.g., "OPENAI_QUERY", "10K_DOCUMENT_QUERY").

        Returns:
            str: A model-generated response grounded in retrieved documents, or an error message.
        """

        try:
            # Ensure that the provided action is valid
            if action not in self.collections:
                return "Invalid action type for retrieval."

            # Check if clients are available
            if self.client is None:
                return "Error: Qdrant client not available"
            
            if self.openaiclient is None:
                return "Error: Azure OpenAI client not available"

            # Step 1: Convert the user query into a dense vector (embedding)
            try:
                query = self.get_text_embeddings(user_query)
            except Exception as embed_err:
                return f"Embedding error: {embed_err}"  # Fail early if embedding fails

            # Step 2: Retrieve top-matching chunks from the relevant Qdrant collection
            try:
                text_hits = self.client.query_points(
                    collection_name=self.collections[action],  # Choose the right collection based on routing
                    query=query,                          # The embedding of the user's query
                    limit=3,                              # Fetch top 3 relevant chunks
                    with_payload=True                     # Include metadata in results
                ).points
            except Exception as qdrant_err:
                return f"Vector DB query error: {qdrant_err}"  # Handle Qdrant access issues

            # Extract the raw content and metadata from the retrieved vector hits
            chunks_with_metadata = []
            for point in text_hits:
                chunk_data = {
                    'content': point.payload.get('content', ''),
                    'document_info': point.payload.get('metadata', {}).get('document_info', 'Unknown source'),
                    'uuid': point.payload.get('metadata', {}).get('uuid', 'No ID')
                }
                chunks_with_metadata.append(chunk_data)

            # If no relevant content is found, return early
            if not chunks_with_metadata:
                return "No relevant content found in the database."

            # Step 3: Pass the retrieved context to the RAG model to generate a response
            try:
                response = self.rag_formatted_response(user_query, chunks_with_metadata)
                return response
            except Exception as rag_err:
                return f"RAG response error: {rag_err}"  # Handle generation failures

        # Catch any unforeseen errors in the overall process
        except Exception as err:
            return f"Unexpected error: {err}"

    def rag_formatted_response(self, user_query: str, chunks_with_metadata: list):
        """
        Generate a response to the user query using the provided context,
        with article references formatted as [1][2], etc., and include source information.
        """
        if self.openaiclient is None:
            return "Error: Azure OpenAI client not initialized"
        
        # Format context with IDs and source information for better referencing
        formatted_context = ""
        source_info = []
        seen_sources = set()  # Track unique document sources
        
        for i, chunk in enumerate(chunks_with_metadata, 1):
            content = chunk.get('content', '')
            document_info = chunk.get('document_info', 'Unknown source')
            
            formatted_context += f"[{i}] {content}\n\n"
            
            # Only add source if we haven't seen this document before
            if document_info not in seen_sources:
                source_info.append(f"[{len(source_info) + 1}] Source: {document_info}")
                seen_sources.add(document_info)
        
        # Create source references section
        sources_section = "\n".join(source_info)
        
        rag_prompt = f"""
You are a helpful assistant that answers questions based solely on the provided context.

Context:
{formatted_context}

User Query: {user_query}

Instructions:
1. Answer the query using ONLY the information provided in the context above
3. If the context doesn't contain enough information to answer the query, say so clearly
4. Be concise but comprehensive in your response

Answer:
"""
        print("Try to generate RAG prompt:")

        try:
            response = self.openaiclient.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": rag_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            # Return JSON with ai_response and sources
            return json.dumps({
                "ai_response": response.choices[0].message.content,
                "sources": sources_section
            })
            
        except OpenAIError as e:
            return f"Azure OpenAI API error: {e}"
        except Exception as e:
            return f"Error generating response: {e}"


