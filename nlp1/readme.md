# Speech Context Chatbot
A chatbot utilizing speech data and storing in vector databases to understand context of the speech collection.

## Overview
This project implements a chatbot capable of understanding context from a collection of speech data. It leverages a vector database for efficient retrieval of relevant speech segments and a language model for generating human-like responses.

## Key Features
Speech Data Integration: Processes and embeds speech data for contextual understanding.
Vector Database: Stores and searches speech embeddings for relevant information.
Language Model: Generates human-like responses based on query and retrieved context.

Installation
```bash
pip install faiss numpy speech_recognition sentence_transformers openai

```

## Usage
1. Prepare Speech Data:
   - Convert speech to text.
   - Clean and preprocess text data.
   - Generate embeddings using sentence_transformers.
2. Create Vector Database:
   - Use Faiss to create an index and populate it with speech embeddings.
3. Implement Chatbot Logic:
   - Create a chatbot class or function to handle query processing, vector database search, and response generation.
4. Integrate Language Model :
   - Use a language model like OpenAI's GPT-3 for improved response generation.

Example Usage
```bash
python main.py
```