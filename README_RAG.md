# MinerU RAG Application Deployment Guide

This guide provides a clear and reliable process for setting up the development environment for the Mineru RAG (Retrieval-Augmented Generation) application on a new machine.

## Prerequisites

- A Unix-like operating system (Linux, macOS, or WSL on Windows).
- `git` installed.
- `conda` (Miniconda or Anaconda) installed.

## Step-by-Step Deployment

Follow these steps to ensure a reproducible and hassle-free setup.

### 1. Clone the Repository

First, clone the project from GitHub to your local machine.

```bash
git clone https://github.com/yingpengma/MinerU.git
cd MinerU
```

### 2. Create and Activate the Conda Environment

We use `conda` to manage our Python environment. This command creates a new environment named `mineru` with Python 3.11.

```bash
conda create --name mineru python=3.11 -y
conda activate mineru
```

### 3. Install All Dependencies

This is the most critical step. We will use the `requirements.txt` file, which is a "perfect snapshot" of the known-to-work development environment. This single command installs all necessary packages with their exact versions.

```bash
pip install -r requirements.txt
```

This process might take a few minutes. Once it's complete, your environment will be an exact replica of the original development setup.

### 4. Run the Application

With the environment fully configured, you can now launch the Streamlit web application.

```bash
streamlit run app.py
```

Your web browser should automatically open a new tab with the RAG application interface. If not, you can manually open the "Local URL" provided in your terminal (e.g., `http://localhost:8501`).
