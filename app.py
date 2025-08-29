from flask import Flask, render_template, request, jsonify
from orchestrator import Orchestrator
import time

app = Flask(__name__)

# Initialize the Orchestrator globally
orchestrator = Orchestrator()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question', '')
    allow_web_search = data.get('allow_web_search', False)
    
    answer, supporting_docs, processing_info, route_info = process_question(question, allow_web_search)
    
    return jsonify({
        'answer': answer,
        'supporting_documents': supporting_docs,
        'processing_info': processing_info,
        'route_info': route_info
    })

def process_question(question, allow_web_search):
    """
    Process the user's question and return answer with supporting documents.
    
    Args:
        question (str): The user's question
        allow_web_search (bool): Whether to allow web search
        
    Returns:
        tuple: (answer, supporting_documents, processing_info, route_info)
    """
    start_time = time.time()
    
    try:
        # Use the orchestrator to process the question
        result = orchestrator.main_orchestrator(question, allow_web_search)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Extract answer and sources from the result
        answer = result.get('answer', 'No answer generated')
        sources = result.get('sources', [])
        
        # Extract processing information with timing
        processing_details = result.get('processing_info', 'No processing information available')
        processing_info = {
            "title": "Processing Details",
            "content": processing_details,
            "processing_time": f"{processing_time:.2f} seconds",
            "sub_queries": result.get('sub_queries', []),
            "reasoning": result.get('reasoning', 'No reasoning provided')
        }
        
        # Extract route information
        route_info = {
            "title": "Query Analysis",
            "content": result.get('final_reasoning', 'No final reasoning provided'),
            "sub_answers": result.get('sub_answers', [])
        }
        
        return answer, sources, processing_info, route_info
        
    except Exception as e:
        # Calculate processing time even for errors
        processing_time = time.time() - start_time
        
        # Fallback response if orchestrator fails
        answer = f"Error processing question: {str(e)}"
        supporting_docs = []
        processing_info = {
            "title": "Processing Error",
            "content": "Failed to process the question using RAG system",
            "processing_time": f"{processing_time:.2f} seconds",
            "sub_queries": [],
            "reasoning": f"Error: {str(e)}"
        }
        route_info = {
            "title": "Error Analysis",
            "content": "Unable to analyze query due to system error",
            "sub_answers": []
        }
        return answer, supporting_docs, processing_info, route_info

if __name__ == '__main__':
    # Use debug=False to avoid multiple processes and Qdrant conflicts
    app.run(debug=False, host='127.0.0.1', port=5000)
