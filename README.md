# Advanced RAG System with Intelligent Query Orchestration

A sophisticated Retrieval-Augmented Generation (RAG) system that intelligently routes queries between multiple data sources and agents to provide comprehensive answers. The system processes 10-K financial documents, OpenAI documentation, and real-time internet content through an intelligent orchestrator.

## 🎯 Key Features

### Intelligent Query Routing
- **Smart Classification**: Automatically classifies queries into OpenAI, 10-K document, or internet search categories
- **Query Decomposition**: Breaks complex queries into sub-queries for comprehensive analysis
- **Multi-Agent Orchestration**: Coordinates multiple specialized agents for optimal results

### Multi-Source Data Processing
- **10-K Financial Documents**: Advanced processing of SEC filings and financial reports
- **OpenAI Documentation**: Comprehensive coverage of OpenAI APIs, tools, and services
- **Real-time Internet Search**: Live web search capabilities via ARES API

### Advanced Embedding & Retrieval
- **Nomic Embeddings**: High-quality text embeddings with 768 dimensions
- **Qdrant Vector Database**: Efficient semantic search and similarity matching
- **Semantic Caching**: Intelligent caching to improve response times and reduce API calls

### AI-Powered Response Generation
- **Azure OpenAI Integration**: GPT-4 powered response generation
- **Context-Aware Responses**: Responses grounded in retrieved documents with source citations
- **Answer Synthesis**: Combines multiple sub-query results into coherent final answers

### Web Interface
- **Flask Web Application**: User-friendly web interface
- **Real-time Processing**: Live query processing with progress indicators
- **Source Attribution**: Clear source references and supporting documents

## 🏗️ System Architecture

```
User Query → Orchestrator → Query Router → Specialized Agents → Response Synthesis
                ↓              ↓               ↓               ↓
         Semantic Cache   Classification   Document/Web     Combined Answer
                                         Retrieval
```

### Core Components

1. **Orchestrator** (`orchestrator.py`)
   - Main coordination layer
   - Query decomposition and routing
   - Answer synthesis and combination

2. **TenkAgent** (`tenk_agent.py`)
   - Handles 10-K document queries
   - OpenAI documentation queries
   - Vector similarity search via Qdrant

3. **InternetAgent** (`internet_agent.py`)
   - Real-time web search capabilities
   - ARES API integration for live content

4. **Semantic Caching** (`semantic_cache.py`)
   - Intelligent query caching
   - Embedding-based similarity matching
   - Performance optimization

5. **Web Application** (`app.py`)
   - Flask-based web interface
   - RESTful API endpoints
   - Real-time query processing

## 🚀 Setup & Installation

### Prerequisites
- Python 3.8+
- Azure OpenAI API access
- ARES API key (for internet search)
- Qdrant vector database

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Configuration
Create a `.env` file with the following variables:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_azure_openai_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# ARES API for Internet Search
ARES_API_KEY=your_ares_api_key

# Optional: Custom paths
QDRANT_DATA_PATH=./qdrant_data
```

### 3. Initialize Vector Database
Ensure your Qdrant collections are set up:
- `10k_data`: For 10-K financial documents
- `opnai_data`: For OpenAI documentation

### 4. Run the Application

#### Web Interface
```bash
python app.py
```
Access the web interface at `http://127.0.0.1:5000`

#### Command Line Usage
```bash
python orchestrator.py
```

## 📚 Usage Examples

### Web Interface
1. Navigate to `http://127.0.0.1:5000`
2. Enter your query in the search box
3. Toggle web search if needed
4. View comprehensive results with source citations

### API Endpoints

#### Ask Question
```http
POST /ask
Content-Type: application/json

{
    "question": "What was Uber's revenue in 2021?",
    "allow_web_search": true
}
```

Response:
```json
{
    "answer": "Detailed answer with analysis...",
    "supporting_documents": [
        {
            "title": "Source 1",
            "content": "Document reference..."
        }
    ],
    "processing_info": {
        "title": "Processing Details",
        "content": "Query decomposed into 2 sub-queries...",
        "processing_time": "2.34 seconds"
    },
    "route_info": {
        "title": "Query Analysis",
        "content": "Multiple sub-queries processed..."
    }
}
```

### Query Types

#### Financial Document Queries
```python
# Examples that route to 10-K documents
"What was Lyft's revenue in 2023?"
"Show me Tesla's operating expenses breakdown"
"Compare revenue growth between companies"
```

#### OpenAI Documentation Queries
```python
# Examples that route to OpenAI docs
"How to fine-tune GPT-4?"
"What are the rate limits for the API?"
"Explain the moderation endpoint"
```

#### Internet Search Queries
```python
# Examples that route to web search
"Latest AI developments in 2024"
"Current stock price of NVIDIA"
"Recent news about machine learning"
```

## 🔧 Configuration Options

### SSL Configuration
The system includes SSL bypass configurations for development environments:
```python
# Automatic SSL certificate handling
os.environ['PYTHONHTTPSVERIFY'] = '0'
ssl._create_default_https_context = ssl._create_unverified_context
```

### Embedding Configuration
```python
# Nomic embedding settings
model='nomic-embed-text-v1.5'
task_type="search_query"
dimensionality=768
```

### Vector Search Parameters
```python
# Qdrant search configuration
limit=3  # Top-k results
with_payload=True  # Include metadata
```

## 📁 Project Structure

```
rag_system/
├── app.py                 # Flask web application
├── orchestrator.py        # Main orchestration logic
├── tenk_agent.py         # Document retrieval agent
├── internet_agent.py     # Web search agent
├── semantic_cache.py     # Intelligent caching system
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables
├── README.md            # Project documentation
├── qdrant_data/         # Vector database storage (excluded from git)
└── templates/           # HTML templates for web interface
    └── index.html
```

## 🛠️ Advanced Features

### Query Decomposition
The system automatically breaks complex queries into manageable sub-queries:
```python
# Complex query example
"Find recent AI advancements and their impact on healthcare and education"

# Automatically decomposed into:
# 1. "What are recent advancements in AI?"
# 2. "What is the impact of AI on healthcare?"
# 3. "What is the impact of AI on education?"
```

### Semantic Caching
Intelligent caching system that:
- Stores query embeddings and responses
- Finds semantically similar cached queries
- Reduces API calls and improves response times

### Source Attribution
Comprehensive source tracking with:
- Document references
- Chunk-level citations
- Source confidence scoring

## 🔍 Troubleshooting

### Common Issues

1. **Qdrant Connection Error**
   ```
   Solution: Ensure Qdrant is running and data path is correct
   ```

2. **Azure OpenAI Authentication**
   ```
   Solution: Verify API key and endpoint in .env file
   ```

3. **SSL Certificate Issues**
   ```
   Solution: System includes automatic SSL bypass for development
   ```

4. **Missing Dependencies**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

## 📊 Performance Metrics

- **Average Query Processing**: 2-5 seconds
- **Embedding Generation**: ~500ms
- **Vector Search**: ~100ms
- **Response Generation**: 1-3 seconds
- **Cache Hit Rate**: 15-30% (improves over time)

## 🔒 Security Considerations

- Environment variables for sensitive API keys
- SSL verification bypass only for development
- Input validation and sanitization
- Rate limiting considerations for production use

## 🚀 Future Enhancements

- [ ] Support for additional document types
- [ ] Enhanced query analytics and logging
- [ ] Multi-language support
- [ ] Advanced caching strategies
- [ ] Real-time document updates
- [ ] User authentication and session management

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review environment configuration
3. Verify API credentials and connectivity
4. Check Qdrant database status

---

**Note**: This system is designed for development and testing. For production deployment, ensure proper security measures, monitoring, and scaling considerations are implemented.
