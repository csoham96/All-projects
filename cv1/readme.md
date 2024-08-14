# Document Summarization API

This project implements a FastAPI-based service that extracts text from image documents and summarizes it using a Large Language Model (LLM). The OCR (Optical Character Recognition) process is handled by PaddleOCR, and the text summarization is performed using a LLaMA-based model.

## Table of Contents

- [Features](#features)
- [Structure](#Structure)
- [Setup and Installation](#setup-and-installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Evaluation](#Evaluation)
- [Logging and Debugging](#logging-and-debugging)
- [Conclusion](#conclusion)

## Features

- **OCR**: Extract text from images using PaddleOCR.
- **Text Summarization**: Summarize the extracted text using a LLaMA-based model.
- **FastAPI**: Serve the summarization service as an API endpoint.
- **GPU Support**: Supports GPU acceleration for both OCR and text summarization.

Structure
```bash
project_root/
├── config.py
├── doc_summarize.py
├── main.py
├── eval.py
├── environment.yml
└── readme.md
```

## Setup and Installation

### Step 1: Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/your-username/doc-summarization-api.git
cd doc-summarization-api
```
### Step 2: Create a Conda Environment

Create and activate a conda environment to manage dependencies:

```bash
conda env create -f environment.yml
```

## Configuration
```bash

use_gpu = True  # Set to False if GPU is not available
device_map = "auto"  # Adjust according to your setup
REPO_ID = "your_llama_model_repository_id"  # Replace with the actual model repository ID

```
For device mapping to run on a small gpu,I have done some experimentations using accelarate library which intialises 0 weights to models different layers 
then copied the most efficient device map into config ,its done by multiple iterations and finding the least device memory possible

This configuration file allows you to control the use of GPU, device mapping, and the LLaMA model repository.

## Usage
### Running the API
Start the FastAPI application using Uvicorn:
```bash
uvicorn main:app --reload
```
###

The API service will be accessible at http://127.0.0.1:8000.

### Example Request
You can use curl or any HTTP client like postman to make requests to the API. Here's an example using curl:

```bash
curl -X POST "http://127.0.0.1:8000/summarize/" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "image=@path_to_your_image_file"
```

## API Endpoints
### POST /summarize/

- **Description**: This endpoint accepts an image file, extracts the text from the image using OCR, and returns a summarized version of the text.
- **Request**: An image file (`image/*`) uploaded via a form-data request.
- **Response**: A JSON object containing the summarized text.

### Example Response:

```json
{
  "summary": "This is the summarized text from the image."
}
```

## Evaluation 
Install library necessary for rouge score evaluation of the output
```bash
pip install rouge_score
```
### Evaluation config
configure your API_URL,image,Ground text summary
Example
```bash
your_image_data="/path/to/dataset/"
reference_summary='example summarised text'
API_URL="http://127.0.0.1:8000/summarize/"
```

To run evaluation 
```bash
python eval.py
```
This will generate average latency of the API's for each image and also rouge score for all the images which have data
## Logging and Debugging

The application uses Python’s logging module to log the progress of model loading and summarization tasks. This is helpful for debugging and monitoring the API.

## Conclusion
This API provides a robust and efficient way to perform OCR and text summarization on image documents. The integration of PaddleOCR and LLaMA offers high performance for these tasks, and the FastAPI framework ensures that the service is scalable and easy to deploy.