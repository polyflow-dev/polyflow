# PolyFlow: Advanced LLM-Powered Data Processing Framework

## Overview
PolyFlow is a sophisticated package(eventually building a smart data platform better than Polars[https://pola.rs/] google sheets, etc) that combines Language Models (LLMs) and vector embeddings for advanced data processing operations. It provides a dataFrame API for semantic operations on structured and unstructured data.

## Core Components

### 1. Model Integration
- **Language Models**
  - GPT-4/3.5 Integration
  - Claude/Anthropic Models
  - Local LLM Support
  - Custom Model Integration

- **Retrieval Models**
  - Base class for vector embeddings and similarity search
  - Implementations:
    - SentenceTransformersRM: Uses sentence-transformers models
    - PolyflowEmbeddingModel: Compatible with LiteLLM-supported models
  - Features:
    - Vector normalization
    - Batch processing
    - Index management

- **Embedding Models**
  - SentenceTransformers
  - OpenAI Embeddings
  - Custom Embedding Models
  - Multi-modal Embeddings (Text, Image, Audio)

- **Neural Networks**
  - PyTorch Integration
  - TensorFlow Support
  - scikit-learn Integration
  - sklearn-compatible API
  - Custom Model Architecture
  - Pre-trained Model Support



### 2. Few examples of PolyFlow Data Operations
For the remaining examples, please refer to the [POLYFLOW_DATA_OPERATION.md](POLYFLOW_DATA_OPERATION.md) file.

#### Vector-Based Operations(Semantic Operations)
- `vector_transform`: Transform data using semantic similarity
  - Configurable thresholds and K-nearest neighbors
  - Support for similarity scoring
  - Custom suffix handling
  
- `vector_index`: Create semantic indices
  - Local index storage
  - Efficient retrieval
  - Compatible with FAISS

- `vector_join`: Semantic dataset joining
  - Natural language join conditions
  - Similarity-based matching
  - Configurable thresholds

- `vector_search`: Semantic search functionality
  - Top-K retrieval
  - Optional reranking
  - Similarity scoring

#### LLM-Based Operations
- `llm_transform`: LLM-powered transformations
  - Structured output formatting
  - Example-based learning
  - Error handling and validation

- `llm_hybrid_classify`: Hybrid classification
  - Combines LLM and embedding models
  - Confidence scoring
  - Model agreement metrics

- `llm_extract`: Information extraction
  - Structured data extraction
  - Template-based extraction
  - JSON output formatting

  