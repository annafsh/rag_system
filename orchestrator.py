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

# Import functions from respective modules
from internet_agent import InternetAgent
from tenk_agent import TenkAgent
from semantic_cache import SemanticCaching

# Load environment variables from .env file
load_dotenv()

# Global singleton instance for TenkAgent
_tenk_agent_singleton = None

def get_tenk_agent_singleton(openaiclient=None):
    """Get or create a singleton TenkAgent instance."""
    global _tenk_agent_singleton
    if _tenk_agent_singleton is None and openaiclient is not None:
        _tenk_agent_singleton = TenkAgent(openaiclient=openaiclient)
    return _tenk_agent_singleton


class Orchestrator:
    """
    A class to orchestrate RAG queries by routing them to appropriate handlers.
    
    This class handles query classification and routing to different agents
    (internet search, document retrieval, etc.) based on query content.
    """
    
    def __init__(self):
        """
        Initialize the Orchestrator with Azure OpenAI client and route mappings.
        """
        # Initialize Azure OpenAI client and API keys
        self.ares_api_key = os.getenv("ARES_API_KEY")
        azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

        if azure_openai_api_key and azure_openai_endpoint:
            self.openaiclient = AzureOpenAI(
                api_key=azure_openai_api_key,
                azure_endpoint=azure_openai_endpoint,
                api_version=azure_openai_api_version
            )
            print("Successfully initialized Azure OpenAI client")
        else:
            print("Warning: Azure OpenAI credentials not found")
            self.openaiclient = None

        # Initialize TenkAgent singleton instance with shared OpenAI client
        self.tenk_agent = get_tenk_agent_singleton(openaiclient=self.openaiclient)

        # Initialize InternetAgent instance
        self.internet_agent = InternetAgent(ares_api_key=self.ares_api_key)

        self.semantic_cache = SemanticCaching()

        # Dictionary that maps the route labels (decided by the router) to their respective functions
        # Each type of query is handled differently:
        # - OPENAI_QUERY and 10K_DOCUMENT_QUERY use document retrieval + RAG
        # - INTERNET_QUERY uses a web search API
        self.routes = {
            "OPENAI_QUERY": self.tenk_agent.retrieve_and_response if self.tenk_agent else None,
            "10K_DOCUMENT_QUERY": self.tenk_agent.retrieve_and_response if self.tenk_agent else None,
            "INTERNET_QUERY": self.internet_agent.get_internet_content,
        }

    def route_query(self, user_query: str, allow_web_search: bool = True):
        """
        Routes user queries to appropriate handlers based on content classification.
        
        Args:
            user_query (str): The user's input query to be classified.
            allow_web_search (bool): Whether to allow internet queries.
            
        Returns:
            dict: Classification result with action, reason, and answer.
        """
        if self.openaiclient is None:
            return {
                "action": "INTERNET_QUERY" if allow_web_search else "OPENAI_QUERY",
                "reason": "Azure OpenAI client not available",
                "answer": ""
            }

        if allow_web_search:
            router_system_prompt = f"""
        As a professional query router, your objective is to correctly classify user input into one of three categories based on the source most relevant for answering the query:
        1. "OPENAI_QUERY": If the user's query appears to be answerable using information from OpenAI's official documentation, tools, models, APIs, or services (e.g., GPT, ChatGPT, embeddings, moderation API, usage guidelines).
        2. "10K_DOCUMENT_QUERY": If the user's query pertains to a collection of documents from the 10k annual reports, datasets, or other structured documents, typically for research, analysis, or financial content.
        3. "INTERNET_QUERY": If the query is neither related to OpenAI nor the 10k documents specifically, or if the information might require a broader search (e.g., news, trends, tools outside these platforms), route it here.

        Your decision should be made by assessing the domain of the query.

        Always respond in this valid JSON format:
        {{
            "action": "OPENAI_QUERY" or "10K_DOCUMENT_QUERY" or "INTERNET_QUERY",
            "reason": "brief justification",
            "answer": "AT MAX 5 words answer. Leave empty if INTERNET_QUERY"
        }}

        EXAMPLES:

        - User: "How to fine-tune GPT-3?"
        Response:
        {{
            "action": "OPENAI_QUERY",
            "reason": "Fine-tuning is OpenAI-specific",
            "answer": "Use fine-tuning API"
        }}

        - User: "Where can I find the latest financial reports for the last 10 years?"
        Response:
        {{
            "action": "10K_DOCUMENT_QUERY",
            "reason": "Query related to annual reports",
            "answer": "Access through document database"
        }}

        - User: "Top leadership styles in 2024"
        Response:
        {{
            "action": "INTERNET_QUERY",
            "reason": "Needs current leadership trends",
            "answer": ""
        }}

        - User: "What's the difference between ChatGPT and Claude?"
        Response:
        {{
            "action": "INTERNET_QUERY",
            "reason": "Cross-comparison of different providers",
            "answer": ""
        }}

        Strictly follow this format for every query, and never deviate.
        User: {user_query}
        """
        else:
            router_system_prompt = f"""
        As a professional query router, your objective is to correctly classify user input into one of two categories based on the source most relevant for answering the query:
        1. "OPENAI_QUERY": If the user's query appears to be answerable using information from OpenAI's official documentation, tools, models, APIs, or services (e.g., GPT, ChatGPT, embeddings, moderation API, usage guidelines).
        2. "10K_DOCUMENT_QUERY": If the user's query pertains to a collection of documents from the 10k annual reports, datasets, or other structured documents, typically for research, analysis, or financial content.

        Note: Internet search is not available, so choose the most appropriate category from the two options above.

        Your decision should be made by assessing the domain of the query.

        Always respond in this valid JSON format:
        {{
            "action": "OPENAI_QUERY" or "10K_DOCUMENT_QUERY",
            "reason": "brief justification",
            "answer": "AT MAX 5 words answer"
        }}

        EXAMPLES:

        - User: "How to fine-tune GPT-3?"
        Response:
        {{
            "action": "OPENAI_QUERY",
            "reason": "Fine-tuning is OpenAI-specific",
            "answer": "Use fine-tuning API"
        }}

        - User: "Where can I find the latest financial reports for the last 10 years?"
        Response:
        {{
            "action": "10K_DOCUMENT_QUERY",
            "reason": "Query related to annual reports",
            "answer": "Access through document database"
        }}

        - User: "Top leadership styles in 2024"
        Response:
        {{
            "action": "OPENAI_QUERY",
            "reason": "General query, using available resources",
            "answer": "Check available documentation"
        }}

        - User: "What's the difference between ChatGPT and Claude?"
        Response:
        {{
            "action": "OPENAI_QUERY",
            "reason": "ChatGPT info available in OpenAI docs",
            "answer": "Refer to OpenAI documentation"
        }}

        Strictly follow this format for every query, and never deviate.
        User: {user_query}
        """

        try:
            # Query the GPT-4 model with the router prompt and user input
            response = self.openaiclient.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": router_system_prompt}]
            )

            # Extract and parse the model's JSON response
            task_response = response.choices[0].message.content
            json_match = re.search(r"\{.*\}", task_response, re.DOTALL)
            json_text = json_match.group()
            parsed_response = json.loads(json_text)
            return parsed_response

        # Handle OpenAI API errors (e.g., rate limits, authentication)
        except OpenAIError as api_err:
            return {
                "action": "OPENAI_QUERY" if not allow_web_search else "INTERNET_QUERY",
                "reason": f"OpenAI API error: {api_err}",
                "answer": ""
            }

        # Handle case where model response isn't valid JSON
        except json.JSONDecodeError as json_err:
            return {
                "action": "OPENAI_QUERY" if not allow_web_search else "INTERNET_QUERY",
                "reason": f"JSON parsing error: {json_err}",
                "answer": ""
            }

        # Catch-all for any other unforeseen issues
        except Exception as err:
            return {
                "action": "OPENAI_QUERY" if not allow_web_search else "INTERNET_QUERY",
                "reason": f"Unexpected error: {err}",
                "answer": ""
            }


    def agentic_rag_for_web(self, user_query: str, allow_web_search: bool = True):
        """
        Web-friendly version of agentic_rag that returns results instead of printing.
        
        Args:
            user_query (str): The user's input question.
            allow_web_search (bool): Whether to allow internet queries.
            
        Returns:
            dict: Dictionary containing answer, route, processing_info, and sources
        """
        try:
            question, embedding, answer, processing_info = self.semantic_cache.ask(user_query)
            if answer:
                return {
                    "answer": answer,
                    "route": "SEMANTIC_CACHE",
                    "processing_info": f"Answer retrieved from semantic cache. {processing_info}",
                    "sources": []
                }
        except Exception as init_err:
            print(f"semantic cache error: {init_err}")

        try:
            # Step 1: Use the router to decide which route the query belongs to
            response = self.route_query(user_query, allow_web_search)

            # Extract the routing decision and the reason behind it
            action = response.get("action", "UNKNOWN")
            reason = response.get("reason", "No reason provided")

            # Step 2: Call the correct function depending on the route
            try:
                route_function = self.routes.get(action)
                if route_function:
                    if action == "INTERNET_QUERY":
                        result = route_function(user_query, action)

                        self.semantic_cache.add_to_cache(question, embedding, result)

                        # Internet queries return plain text
                        return {
                            "answer": result,
                            "route": action,
                            "processing_info": reason,
                            "sources": []
                        }
                    else:
                        # Document queries return JSON with ai_response and sources
                        result = route_function(user_query, action)
                        
                        # Try to parse JSON response
                        try:
                            parsed_result = json.loads(result)
                            ai_response = parsed_result.get("ai_response", result)
                            sources_text = parsed_result.get("sources", "")
                            
                            # Convert sources text to list format for supporting documents
                            sources_list = []
                            if sources_text:
                                for line in sources_text.split('\n'):
                                    line = line.strip()
                                    if line and line.startswith('[') and '] Source:' in line:
                                        # Extract source info: "[1] Source: document_name"
                                        source_info = line.split('] Source: ', 1)
                                        if len(source_info) == 2:
                                            source_num = source_info[0].replace('[', '')
                                            source_name = source_info[1]
                                            sources_list.append({
                                                "title": f"Source {source_num}",
                                                "content": source_name
                                            })

                            # Add to cache
                            self.semantic_cache.add_to_cache(question, embedding, ai_response)
                            
                            return {
                                "answer": ai_response,
                                "route": action,
                                "processing_info": reason,
                                "sources": sources_list
                            }
                            
                        except json.JSONDecodeError:
                            # Fallback if JSON parsing fails
                            return {
                                "answer": result,
                                "route": action,
                                "processing_info": reason,
                                "sources": []
                            }
                else:
                    raise ValueError(f"Unsupported action: {action}")
            except Exception as exec_err:
                result = f"Execution error: {exec_err}"
                return {
                    "answer": result,
                    "route": action,
                    "processing_info": reason,
                    "sources": []
                }

        except Exception as err:
            return {
                "answer": f"Unexpected error occurred: {err}",
                "route": "ERROR", 
                "processing_info": "System error",
                "sources": []
            }

    def main_orchestrator(self, user_query: str, allow_web_search: bool = True):
        """
        Main orchestrator that breaks complex queries into sub-queries, processes each one,
        and combines the results into a comprehensive answer.
        
        Args:
            user_query (str): The user's input question.
            allow_web_search (bool): Whether to allow internet queries.
            
        Returns:
            dict: Dictionary containing the final combined answer and processing details
        """
        try:
            # Step 1: Break the query into sub-queries
            sub_query_result = self.sub_queries(user_query)
            
            if not sub_query_result:
                return {
                    "answer": "Error: Failed to process query decomposition",
                    "sub_queries": [],
                    "sub_answers": [],
                    "reasoning": "Query decomposition failed",
                    "processing_info": "Query decomposition failed",
                    "sources": []
                }
            
            sub_queries_list = sub_query_result.get("sub_queries", [])
            reasoning = sub_query_result.get("reasoning", "")
            
            # Step 2: If no sub-queries, treat the original query as a single sub-query
            if not sub_queries_list:
                print(f"Processing original query directly: {reasoning}")
                sub_queries_list = [user_query]
                final_reasoning = "Single query processed directly"
                processing_info = "Single query processed directly"
            else:
                print(f"Processing {len(sub_queries_list)} sub-queries...")
                final_reasoning = "Multiple sub-queries processed and combined"
                processing_info = f"Query decomposed into {len(sub_queries_list)} sub-queries"
            
            # Step 3: Process each sub-query (unified for both single and multiple queries)
            sub_answers = []
            all_sources = []
            processing_details = []
            
            for i, sub_query in enumerate(sub_queries_list, 1):
                print(f"Processing sub-query {i}: {sub_query}")
                try:
                    sub_result = self.agentic_rag_for_web(sub_query, allow_web_search)
                    sub_processing_info = sub_result.get("processing_info", "No processing info provided")
                    processing_details.append(f"Sub-query {i}: {sub_processing_info}")
                    
                    sub_answers.append({
                        "query": sub_query,
                        "answer": sub_result.get("answer", "No answer generated"),
                        "route": sub_result.get("route", "Unknown"),
                        "processing_info": sub_processing_info,
                        "sources": sub_result.get("sources", [])
                    })
                    
                    # Collect all sources
                    sources = sub_result.get("sources", [])
                    for source in sources:
                        # Add sub-query context to source title
                        source_with_context = {
                            "title": f"Sub-query {i}: {source.get('title', 'Unknown Source')}",
                            "content": source.get('content', 'No content')
                        }
                        all_sources.append(source_with_context)
                        
                except Exception as sub_err:
                    print(f"Error processing sub-query {i}: {sub_err}")
                    error_info = f"Processing failed: {sub_err}"
                    processing_details.append(f"Sub-query {i}: {error_info}")
                    
                    sub_answers.append({
                        "query": sub_query,
                        "answer": f"Error processing sub-query: {sub_err}",
                        "route": "ERROR",
                        "processing_info": error_info,
                        "sources": []
                    })
            
            # Combine processing details
            combined_processing_info = f"{processing_info}. " + "; ".join(processing_details)
            
            # Step 4: Combine all answers (or return single answer if only one sub-query)
            try:
                if len(sub_answers) == 1:
                    # For single queries, return the answer directly without combination
                    combined_answer = sub_answers[0]["answer"]
                    final_sources = sub_answers[0]["sources"]
                else:
                    # For multiple sub-queries, combine using OpenAI
                    combined_answer = self.combine_answers(user_query, sub_answers)
                    final_sources = all_sources                    
            except Exception as combine_err:
                print(f"Error combining answers: {combine_err}")
                # Fallback: concatenate answers
                combined_answer = self.fallback_combine_answers(sub_answers)
                final_sources = all_sources               
            
            return {
                "answer": combined_answer,
                "sub_queries": sub_query_result.get("sub_queries", []),
                "sub_answers": sub_answers,
                "reasoning": reasoning,
                "final_reasoning": final_reasoning,
                "processing_info": combined_processing_info,
                "sources": final_sources
            }
            
        except Exception as err:
            print(f"Error in main_orchestrator: {err}")
            return {
                "answer": f"Error in orchestration: {err}",
                "sub_queries": [],
                "sub_answers": [],
                "reasoning": "Orchestration failed",
                "processing_info": f"Orchestration failed: {err}",
                "sources": []
            }

    def combine_answers(self, original_query: str, sub_answers: list):
        """
        Combine multiple sub-query answers into a comprehensive response using OpenAI.
        
        Args:
            original_query (str): The original user query
            sub_answers (list): List of sub-query results
            
        Returns:
            str: Combined comprehensive answer
        """
        if self.openaiclient is None:
            return self.fallback_combine_answers(sub_answers)
        
        # Format the sub-answers for the prompt
        formatted_sub_answers = ""
        for i, sub_answer in enumerate(sub_answers, 1):
            formatted_sub_answers += f"""
Sub-query {i}: {sub_answer['query']}
Answer: {sub_answer['answer']}
Source: {sub_answer['route']}

"""
        
        combine_prompt = f"""
You are tasked with combining multiple answers from sub-queries into a single, comprehensive response to the original user question.

Original Question: {original_query}

Sub-query Results:
{formatted_sub_answers}

Instructions:
1. Synthesize the information from all sub-answers into a coherent, comprehensive response
2. Ensure the final answer directly addresses the original question
3. If there are contradictions between sub-answers, note them appropriately
4. If some sub-answers contain errors or are not relevant, focus on the useful information
5. Maintain a natural, flowing narrative while incorporating all relevant information
6. If the sub-answers don't fully address the original question, acknowledge any gaps

Provide a well-structured, comprehensive answer that combines all relevant information:
"""
        
        try:
            response = self.openaiclient.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": combine_prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error in OpenAI combination: {e}")
            return self.fallback_combine_answers(sub_answers)

    def fallback_combine_answers(self, sub_answers: list):
        """
        Fallback method to combine answers when OpenAI is not available.
        
        Args:
            sub_answers (list): List of sub-query results
            
        Returns:
            str: Simple concatenated answer
        """
        if not sub_answers:
            return "No answers to combine."
        
        combined = "Based on the analysis of multiple aspects of your question:\n\n"
        
        for i, sub_answer in enumerate(sub_answers, 1):
            if sub_answer['answer'] and not sub_answer['answer'].startswith('Error'):
                combined += f"{i}. Regarding '{sub_answer['query']}':\n"
                combined += f"   {sub_answer['answer']}\n\n"
        
        if combined == "Based on the analysis of multiple aspects of your question:\n\n":
            combined = "I encountered difficulties processing your question. Please try rephrasing or breaking it down differently."
        
        return combined

    def sub_queries(self, user_query: str):
        sub_query_prompt = f"""
        Break a query into smaller, logical sub-queries when applicable, in order to simplify solving or analyzing complex requests.

For each query, assess whether it can be divided into independent, smaller sub-queries that address each component of the request. If division is not applicable, return the original query as-is.

# Steps

1. **Assess Complexity**: Analyze whether the query is simple or complex. A complex query is one that poses multiple questions, includes conjunctions (e.g., "and," "or"), or involves multiple aspects to evaluate or solve.  
2. **Divide into Sub-queries** (if applicable):  
   - Identify distinct components or tasks within the query.  
   - For each component, create a standalone sub-query that can be addressed independently.  
   - Maintain logical order and preserve meaning while restructuring into sub-queries.  
3. **Return Query As-Is** (if not applicable): If no subdivisions are possible, return the query in its original form while clearly stating that no sub-queries were necessary.  

# Output Format

Return a JSON object with the following fields:  
- "original_query": The original query provided.  
- "sub_queries": A list of sub-queries if the query was divided, or an empty list if no sub-queries are applicable.  
- "reasoning": A brief explanation (1-2 sentences) that justifies why the query was or was not divided into sub-queries.

Example format:

```json
{{
  "original_query": "query text here",
  "sub_queries": ["sub-query 1", "sub-query 2", "..."],
  "reasoning": "reasoning text here"
}}
```

# Examples

### Example 1 (Complex Query with Sub-queries)  
**Input**:  
"Find recent advancements in AI and their impact on healthcare and education."  

**Output**:  
```json
{{
  "original_query": "Find recent advancements in AI and their impact on healthcare and education.",
  "sub_queries": [
    "What are recent advancements in AI?",
    "What is the impact of recent advancements in AI on healthcare?",
    "What is the impact of recent advancements in AI on education?"
  ],
  "reasoning": "The original query involves multiple distinct aspects: advancements in AI and their specific impacts on two domains (healthcare and education). Breaking it down allows focused answers for each part."
}}
```

### Example 2 (Simple Query with No Sub-queries)  
**Input**:  
"What is the capital of France?"  

**Output**:  
```json
{{
  "original_query": "What is the capital of France?",
  "sub_queries": [],
  "reasoning": "The query is simple and contains only one aspect, so no sub-queries are necessary."
}}
```

### Example 3 (Nested Questions with Sub-queries)  
**Input**:  
"Explain how photosynthesis works and why it is important for agriculture and the ecosystem."  

**Output**:  
```json
{{
  "original_query": "Explain how photosynthesis works and why it is important for agriculture and the ecosystem.",
  "sub_queries": [
    "How does photosynthesis work?",
    "Why is photosynthesis important for agriculture?",
    "Why is photosynthesis important for the ecosystem?"
  ],
  "reasoning": "The query consists of one primary scientific explanation and two distinct subtopics related to its importance. Dividing it helps in addressing each aspect comprehensively."
}}
```

# Notes

- Always ensure that sub-queries retain logical flow and full coverage of the original query's intent.  
- When dividing, focus on creating fully independent and answerable sub-queries wherever possible.  
- Avoid trivial splits (e.g., "Find X and explain Y" being split into "Find X" and "Explain Y" unless necessary).

Query: "{user_query}"
Output:
        """
        try:
            response = self.openaiclient.chat.completions.create(
                  model="gpt-4o",
                  messages=[
                      {"role": "system", "content": sub_query_prompt},
                  ]
              )
            
            # Extract and parse the model's JSON response
            task_response = response.choices[0].message.content
            json_match = re.search(r"\{.*\}", task_response, re.DOTALL)
            if json_match:
                json_text = json_match.group()
                parsed_response = json.loads(json_text)

                print(parsed_response)
                return parsed_response
            else:
                # Fallback if no JSON found
                print(f"Failed to parse response - treating as single query")
                return {
                    "original_query": user_query,
                    "sub_queries": [],
                    "reasoning": "Failed to parse response - treating as single query"
                }
                
        except Exception as err:
            print(f"Error processing query: {err}")
            return {
                "original_query": user_query,
                "sub_queries": [],
                "reasoning": f"Error processing query: {err}"
            }

# Example usage when running this file directly
if __name__ == "__main__":
    # Create an instance of Orchestrator
    orch = Orchestrator()

    print(orch.sub_queries("what is revenue of lyft and what is reveue for uber in 2024?"))

    # Test the route_query function
    result = orch.route_query("what is the revenue of uber in 2021?")
    print(result)
    
    # You can also test other queries
    result2 = orch.route_query("How to fine-tune GPT-4?")
    print(result2)

    result3 = orch.route_query("How to setup auth with Okta?")
    print(result3)

    # Test the full agentic RAG pipeline
    orch.agentic_rag("what was uber revenue in 2021?")
    orch.agentic_rag("what was lyft revenue in 2024?")
    orch.agentic_rag("List me down new LLMs in 2025")
    orch.agentic_rag("how to work with chat completions?")