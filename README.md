# Search Showdown: BM25 vs. Neural Search under Vocabulary Mismatch

**Course:** CS F469: Information Retrieval  
**Group:** 4  

## Project Overview

This repository contains a comprehensive Information Retrieval (IR) study and an interactive web application designed to evaluate and visualize the performance of different search paradigms under "vocabulary mismatch" conditions. 

Classical lexical search engines often fail when a user's query and a relevant article describe the same event using different terminology. While dense neural search engines can bridge these semantic gaps, they present their own challenges, such as negation blindness and homonym confusion. This project empirically compares these approaches using the **HuffPost News Category Dataset v3**.

## System Architecture

The project consists of two main components: a rigorous Python backend for large scale evaluation and an interactive client side web application for live demonstration.

### 1. Backend Evaluation Pipeline (`Revised_IR.ipynb`)
The Jupyter Notebook implements a from-scratch IR pipeline evaluating three core engines across 209,527 news articles:
*   **Engine A (Custom BM25):** A pure Python/NumPy implementation of the BM25 algorithm featuring a custom negation-preserving text normalizer, exact Lucene IDF scoring, and an optional Porter Stemmer.
*   **Engine B (Dense Neural Retrieval):** Utilizes the `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` model to encode articles into 384-dimensional vectors, indexed using `faiss.IndexFlatIP` for exact cosine similarity search.
*   **Engine C (Hybrid RRF):** Combines the results of Engine A and Engine B using Reciprocal Rank Fusion (RRF) to leverage the strengths of both lexical and semantic search.

The pipeline includes a rule-based evaluation suite across five adversarial query families (Negation, Homonym, Temporal, Vocabulary Mismatch, and Exact Name) and executes advanced diagnostics including Lexical Coverage stratification and Oracle Pseudo-Relevance Feedback (PRF).

### 2. Interactive Web Application (`Search Showdown Playground.html`)
The frontend is a standalone, client side web application that visualizes the mathematical mechanics of the search engines. It requires no backend server.
*   **Live Search:** Runs a quantized ONNX version of the MiniLM model directly in the browser via Transformers.js, allowing real time semantic and lexical search against a loaded sample corpus.
*   **Word Match Lab:** An interactive tool to calculate lexical overlap (Coverage) between custom queries and articles.
*   **BM25 Lab:** A visual calculator demonstrating the impact of Term Frequency (TF), Inverse Document Frequency (IDF), and document length on the BM25 scoring curve.
*   **Dense Lab:** Visualizes vector representations and calculates cosine similarity.
*   **Hybrid Lab & Mixer:** Interactive interfaces to experiment with Reciprocal Rank Fusion weights and smoothing constants.

## Installation and Setup

### Backend Execution
1.  Ensure you have Python 3.10+ installed.
2.  Install the required dependencies:
    ```bash
    pip install sentence-transformers faiss-cpu scipy nltk pandas numpy matplotlib torch
    ```
3.  Download the **HuffPost News Category Dataset v3** from Kaggle and place `News_Category_Dataset_v3.json.zip` in the root directory.
4.  Execute `Revised_IR.ipynb` in Jupyter Notebook or Google Colab. The script will automatically build the indices, run the evaluations, and generate CSV reports and summary plots.

### Frontend Execution
1.  Open `Search Showdown Playground.html` in any modern web browser.
2.  For the **Live Search** and **Dense Lab** features, an active internet connection is required on the first load to download the browser-compatible MiniLM model (~23 MB).
3.  To test the Live Search with custom data, you can upload the unzipped `News_Category_Dataset_v3.json` file directly through the interface to extract a testing sample.

## Key Findings

Based on the evaluation of 288 rule-graded relevant pairs:
*   **Vocabulary Mismatch:** Dense retrieval significantly outperforms BM25 when queries and documents share few exact words, maintaining a 44.44% Recall@100 even at zero lexical overlap.
*   **Precision vs. Recall:** Engine B (Dense) achieved the highest overall top-rank precision, while Engine C (Hybrid RRF) achieved the highest overall Recall@100.
*   **Adversarial Weaknesses:** Dense models struggle significantly with explicit negation constraints and temporal filtering, pulling a higher rate of hard-negative documents into the top results compared to BM25.
*   **Efficiency:** The custom BM25 implementation proved highly efficient, requiring approximately 11x less memory and executing queries nearly 20x faster than the dense FAISS index.
