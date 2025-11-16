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

### 1. Download prompts: Download the HarmBench test set from the HarmBench GitHub repository and upload the resulting CSV file to your GCS bucket.

### 2. Upload data to Google Cloud Storage: 
- Upload the C4 data subset (e.g., for finding glitch tokens) to your GCS bucket.
- Upload the downloaded HarmBench prompts CSV to your GCS bucket.

Example GCS structure:

gs://your-bucket/
├── c4_subset.csv
├── harmbench_prompts.csv
└── outputs/
---

## Configuration


