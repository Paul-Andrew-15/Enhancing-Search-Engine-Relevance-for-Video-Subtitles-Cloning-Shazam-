# Enhancing-Search-Engine-Relevance-for-Video-Subtitles-Cloning-Shazam-

## Project Overview
This project focuses on developing AI-driven search systems, audio-to-text conversion, and interactive search applications. It integrates advanced search engine technologies, vector embeddings, speech-to-text models, and subtitle processing to enhance information retrieval and media analysis.

## Objective
The goal is to build an intelligent search engine that retrieves subtitles efficiently based on user queries. By leveraging Natural Language Processing (NLP) and machine learning, the system improves search relevance and accuracy.

## Keyword-Based vs. Semantic Search
1. Keyword-Based Search: Matches user queries with indexed documents based on exact keywords.
2. Semantic Search: Understands the meaning and context of queries beyond keyword matching for improved relevance.

## Workflow
The search engine processes user queries and compares them against subtitle documents through the following steps:

1. Data Preprocessing
   Extract and clean subtitle data by removing timestamps, punctuation, and unnecessary elements.
   Use a subset of the dataset (e.g., 30%) for computational efficiency.

2. Text Vectorization
   Convert subtitle text into vector embeddings using:
   Bag of Words (BoW) / TF-IDF for keyword-based search.
   BERT-based SentenceTransformers for semantic search.

3. Document Chunking
   Divide large subtitle documents into smaller text segments while maintaining context.
   Implement overlapping windows to prevent information loss.

4. Storing Embeddings in ChromaDB
   Store vector representations of subtitle chunks in ChromaDB for fast and efficient retrieval.

5. Query Processing & Retrieval
   Convert user audio queries into text using speech recognition models like AssemblyAI.
   Vectorize the text query and compute cosine similarity with stored subtitle embeddings.
   Rank and return the most relevant subtitle segments.

## Project Components & Notebooks
1. Audio_2_Text.ipynb:
   Converts audio files to text using speech recognition models (e.g., AssemblyAI).
   Useful for transcriptions, podcasts, and accessibility applications.
2. Chroma_db_Embeddings_V2.ipynb:
   Implements ChromaDB for vector embedding storage and management.
   Enables efficient similarity search, document indexing, and AI-powered retrieval.
3. Gradio_Search_Engine_Demo.ipynb:
   Builds an interactive search engine demo using Gradio.
   Supports keyword-based, vector-based, and hybrid search methods.
4. Search_Engine_Extracting_Data.ipynb:
   Handles data extraction, web scraping, and indexing for structured document retrieval.
   Prepares text for efficient tokenization and search indexing.
5. Shazam_Clone_Search_Engine.ipynb:
   Develops an audio recognition system similar to Shazam.
   Identifies music, speech, or sound patterns from audio samples.
6. Subtitles_Chunking.ipynb:
   Splits subtitle files into smaller segments for better indexing and retrieval.
   Supports automatic subtitle generation, video search indexing, and multilingual translation.
7. Testing_Search_Mechanism.ipynb:
   Evaluates different search algorithms, including BM25, TF-IDF, and vector-based retrieval.
   Assesses ranking efficiency for improved search results.

## Libraries Used
1. gradio – For interactive UI development.
2. assemblyai – For speech-to-text conversion.
3. pydub – For audio processing.
4. python-dotenv – For environment variable management.
5. chromadb – For vector database storage.
6. sentence-transformers – For generating text embeddings.

## Applications
1. Multimodal Search Engines – Combining text, audio, and embeddings for advanced search.
2. Audio & Music Recognition – Identifying songs and sound patterns using AI.
3. Semantic Search & NLP – Implementing vector-based document retrieval.
4. Interactive Search Interfaces – Using Gradio to create user-friendly search applications.
5. Video & Subtitle Processing – Extracting and structuring subtitles for accessibility and indexing.

## Summary
This project integrates state-of-the-art AI search technologies, speech recognition, and NLP techniques to enhance media retrieval and transcription. By leveraging vector databases, machine learning, and advanced search ranking models, it provides powerful tools for intelligent search, media indexing, and interactive search applications.
