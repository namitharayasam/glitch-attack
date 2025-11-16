# Glitch Token Analysis

A pipeline for analyzing language model safety vulnerabilities using **glitch tokens**. This project tests whether specific, context-sparse tokens can bypass safety mechanisms in large language models by comparing model responses with and without these injected tokens.

---

## Overview

This repository contains three main Python scripts that form a complete analysis pipeline:

* **`inference.py`** - Generates model responses with pure prompts as well as injected glitch tokens at various positions (prefix, middle, suffix).
* **`classify.py`** - Classifies generated responses using the HarmBench Llama-2-13b classifier to determine if safety mechanisms were bypassed.
* **`asr_calc.py`** - Calculates the **Attack Success Rate (ASR)** by comparing pure vs. glitched prompt performance.

---

## Features

* Support for multiple language models via HuggingFace
* Configurable glitch token injection (prefix, middle, suffix positions)
* Batch processing with GPU support
* Google Cloud Storage (GCS) integration for data I/O
* Detailed ASR metrics and comparison

---

## Prerequisites

* **Python 3.10+**
* **CUDA-capable GPU** (recommended for performance)
* **Google Cloud Platform** account with Storage access
* **HuggingFace access for gated models**

---

## Installation

### 1. Clone the repository:

```bash
git clone [https://github.com/yourusername/glitch-token-analysis.git](https://github.com/yourusername/glitch-token-analysis.git)
cd glitch-token-analysis
```
### 2. Install dependencies:
```bash
pip install -r requirements.txt
```
### 3. Set up credentials:
```bash
cp .env.example .env
# Edit .env with your actual credentials
```
---

## Data Preparation

### 1. Download prompts: 
Download the HarmBench test set from the [HarmBench GitHub repository](https://github.com/centerforaisafety/HarmBench) and upload the resulting CSV file to your GCS bucket.

### 2. Upload data to Google Cloud Storage: 
- Upload the C4 data subset (e.g., for finding glitch tokens) to your GCS bucket.
- Upload the downloaded HarmBench prompts CSV to your GCS bucket.

Example GCS structure:

- `gs://your-bucket/`
  - `c4_subset.csv`
  - `harmbench_prompts.csv`
  - `outputs/`
---

## Configuration
- Edit the glitch token list in classify.py
- Change the `ENTRYPOINT` in Dockerfile based on which script you want to run


## Usage
### Running on Google Cloud Vertex AI

1. Build Docker image:
```bash
docker build --platform linux/amd64 -t gcr.io/your-project-id/glitch-analysis:latest .
```

2. Push to Google Container Registry:

```bash
docker push gcr.io/your-project-id/glitch-analysis:latest
```

3. Submit training job:
```bash
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=glitch-analysis-job \
  --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,container-image-uri=gcr.io/your-project-id/glitch-analysis:latest
```

4. Adjust the GPU based on requirements:

- For larger models: Use `NVIDIA_TESLA_V100` or `NVIDIA_TESLA_A100`
- For more memory: Increase machine-type to `n1-highmem-8` or `n1-highmem-16`
- For faster processing: Increase accelerator-count or use multiple workers

---
### Arguments

- `--model-name` - HuggingFace model name (ex: `mistralai/Mistral-7B-Instruct-v0.2`)  
- `--bucket` - GCS bucket name  
- `--input-gcs` - Input CSV path in GCS  
- `--output-gcs` - Output CSV path in GCS  
- `--glitch-tokens` - List of glitch tokens to test  
- `--positions` - Token positions: prefix, middle, suffix  
- `--num-examples` - Number of examples to process (default: all)  
- `--batch-size` - Batch size (default: 32)  
- `--max-new-tokens` - Max tokens to generate (default: 100)

## Docker build fails:

- Ensure platform is set: ```bash --platform linux/amd64 ```
- Check Docker has enough memory allocated
---

## Output Example
### inference.py
```csv
prompt_id,prompt,glitch_token,position,response
0,"Prompt1",None,pure,"I cannot help with that..."
1,"Prompt2","glitchtoken",suffix,"Here's how to..."
```
### classify.py 
```csv
prompt_id,prompt,glitch_token,position,response,prediction
0,"Prompt1",None,pure,"I cannot...",No
1,"Prompt2","glitchtoken",suffix,"Here's how...",Yes
```

