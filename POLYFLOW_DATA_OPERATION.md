# Polyflow Vector Semantic Operations

This document outlines the key DataFrame accessors and operations available in the Polyflow library for vector semantic processing.

## DataFrame Accessors

### Time Series Analysis
- `vecllm_time_series`: Time series analysis with semantic understanding
  - `detect_anomalies()`: Detect anomalies in time series data
  - `forecast()`: Generate semantic time series forecasts
  - `find_patterns()`: Find patterns matching semantic descriptions

### Classification Operations
- `np_llm_hybrid_classify`: NumPy-based hybrid classification using LM and embeddings
- `llm_hybrid_classify`: Hybrid classification combining LM and embedding models
- `llm_classify`: Pure LM-based semantic classification
  ```python
  df.llm_classify(text_column="text", categories=["A", "B", "C"])
  ```

### Clustering and Sorting
- `sk_llm_cluster`: Semantic clustering with scikit-learn integration
  ```python
  df.sk_llm_cluster(text_column="description", n_clusters=5)
  ```
- `llm_topk`: Semantic top-K sorting using various algorithms
  ```python
  df.llm_topk(user_instruction="Sort by relevance", K=10)
  ```

### Transformation and Mapping
- `llm_transform`: LLM-based semantic transformations
  ```python
  df.llm_transform(column="text", transform_query="Extract key topics")
  ```

### Vector Operations
- `vector_join`: Semantic vector join operations
  ```python
  df1.vector_join(df2, join_instruction="Match similar items")
  ```
- `vector_transform`: accessor provides powerful vector-based transformations:
    ```python
    df.vector_transform(column="description",transform_query="technical content about AI", K=5, threshold=0.7, return_scores=True)
    ```
- `vector_index`: accessor manages semantic indices:
    ```python
    # Create an index
    df = df.vector_index("text_column", "index_directory")
    ```
- `vector_search`: accessor enables semantic search:
    ```python
    results = df.vector_search(
        col_name="description",
        query="machine learning frameworks",
        K=10,
        n_rerank=5,
        return_scores=True
    )
    ```
- `vector_index`: creates persistent index storage and generate vector embeddings
    ```python
    # Create semantic index
    df = df.vector_index("text", "index_path")
    ```

### Semantic Extraction
- `llm_extract`: accessor extracts structured information:

    ```python
    extracted = df.llm_extract(
        input_cols=["text"],
        output_cols={
            "topic": "Main topic of discussion",
            "sentiment": "Sentiment of the text",
            "key_points": "Key points mentioned"
        },
        extract_quotes=True
    )
    ```

## Key Features

### Hybrid Processing
- Combines LLM and embedding models for improved accuracy
- Configurable confidence thresholds and model cascading
- Support for both semantic and vector-based operations

### Performance Optimizations
- NumPy-based implementations for efficiency
- Parallel processing support for grouped operations
- Cascading model architecture to balance speed and accuracy

### Flexibility
- Multiple sorting and clustering algorithms
- Customizable prompts and templates
- Support for various input formats and data types

### Integration
- Seamless pandas DataFrame integration
- Compatible with scikit-learn workflows
- Support for various LLM and embedding models

## Usage Examples

### Time Series Analysis
```python
# Detect anomalies
df.vecsem_time_series.detect_anomalies(
    time_col="timestamp",
    value_col="price",
    description="Detect unusual price movements"
)

# Generate forecasts
df.vecsem_time_series.forecast(
    time_col="date",
    value_col="sales",
    horizon=7
)
```

### Classification
```python
# Hybrid classification
df.sem_hybrid_classify(
    text_column="description",
    categories=["urgent", "normal", "low"],
    use_embeddings=True
)
```

### Vector Operations
```python
# Semantic join
result = df1.vector_join(
    df2,
    join_instruction="Match products with similar descriptions",
    return_explanations=True
)
```

## Configuration

Most operations require configuring the language model and/or embedding model:

```python
import polyflow

polyflow.settings.configure(
    lm="your-language-model",
    rm="your-embedding-model"
)
```

## Best Practices

1. Use hybrid approaches when accuracy is critical
2. Enable cascading for large-scale operations
3. Configure appropriate thresholds for your use case
4. Consider using parallel processing for grouped operations
5. Monitor and adjust confidence scores as needed