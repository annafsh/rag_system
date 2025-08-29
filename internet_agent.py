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


class InternetAgent:
    """
    A class to handle internet-based queries using ARES API for live search.
    
    This class encapsulates the functionality for fetching real-time information
    from the internet when queries require current data beyond internal datasets.
    """

    def __init__(self, ares_api_key: str):
        """
        Initialize the InternetAgent with ARES API key.
        """
        self.ares_api_key = ares_api_key  # Load ARES API key from environment

    def get_internet_content(self, user_query: str, action: str):
        """
        Fetches a response from the internet using ARES-API based on the user's query.

        This function serves as the tool invoked when the router classifies a query
        as requiring real-time information beyond internal datasets—i.e., "INTERNET_QUERY".
        It sends the query to a live search API (ARES) and returns the result.

        Args:
            user_query (str): The user's question that needs a live answer.
            action (str): Route type (always expected to be "INTERNET_QUERY").

        Returns:
            str: Response text generated using internet search or an error message.
        """
        print("Getting your response from the internet 🌐 ...")

        #return "Some dummy response from the internet"

        # API endpoint for the ARES live search tool
        url = "https://api-ares.traversaal.ai/live/predict"

        # Payload structure expected by the ARES API
        payload = {"query": [user_query]}

        # Authentication and content headers for API access
        headers = {
            "x-api-key": self.ares_api_key,  # Your secret API key (should be securely loaded from environment)
            "content-type": "application/json"
        }

        try:
            # Send the query to the ARES API and check for success
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()

            # Extract and return the main response text from the API's nested JSON
            return response.json().get('data', {}).get('response_text', "No response received.")

        # Handle HTTP-level errors (e.g., 400s or 500s)
        except requests.exceptions.HTTPError as http_err:
            return f"HTTP error occurred: {http_err}"

        # Handle general connection, timeout, or request formatting issues
        except requests.exceptions.RequestException as req_err:
            return f"Request error occurred: {req_err}"

        # Catch-all for any unexpected failure
        except Exception as err:
            return f"An unexpected error occurred: {err}"


