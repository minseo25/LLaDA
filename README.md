# LLaDA: Quick Start Guide

This guide explains how to set up and run the LLaDA project, including the chat interface, web demo, and evaluation scripts.

---

## 1. Create and Activate a Virtual Environment (Recommended)

```bash
python3.12 -m venv .venv
source .venv/bin/activate  # On Linux/Mac
```

---

## 2. Install Dependencies

Make sure you have the latest pip:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Set Environment Variables (for evaluation and code execution)

Some evaluation tasks require code execution and remote dataset loading. Set the following environment variables:

```bash
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_HOME=/rscratch/minseokim
export HF_HUB_CACHE=/rscratch/minseokim/hub
export HF_DATASETS_CACHE=/rscratch/minseokim/datasets
export TRANSFORMERS_CACHE=/rscratch/minseokim/transformers
```

---

## 4. Run the Profiling

This will perform a runtime profiling of LLaDA-8B-Instruct and LLaMA-3-8B-Instruct.

```bash
python run_llm_runtime_profile.py
```

---

## 5. Run the Chat Interface (Terminal)

This allows you to interact with the LLaDA-8B-Instruct model in the terminal.

```bash
python chat.py
```

You will be prompted to enter your question, and the model will reply in the terminal.

---

## 6. Run the Web Demo (Gradio)

This launches a web-based chat demo using Gradio.

```bash
python app.py
```

After running, a local URL (e.g., http://127.0.0.1:7860) will be displayed. Open it in your browser to chat with the model.

---

## 7. Run Evaluation Scripts

To reproduce benchmark results, use the provided shell script:

```bash
bash eval_llada.sh
```

This will run a series of evaluation tasks as described in the paper. Make sure the required environment variables are set (see step 3).

