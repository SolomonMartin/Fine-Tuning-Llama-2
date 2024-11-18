# Fine-Tuning Llama 2 Model

This repository provides a solution for fine-tuning the **Llama 2** model on custom datasets using the Hugging Face Transformers library. It includes the necessary code for dataset preparation, model fine-tuning, and inference generation. The goal of this project is to leverage the power of the Llama 2 architecture to fine-tune and adapt it to specific tasks, enabling high-quality text generation.

## Project Description

The primary goal of this project is to fine-tune the **Llama 2** model, a state-of-the-art language model, on your own dataset. Fine-tuning allows the model to learn specific domain knowledge, making it more suited to particular use cases. For this project, you’ll be training the model using the Hugging Face `transformers` library and the `datasets` library, along with tools like `accelerate` for distributed training.

The project includes the following components:
- **Fine-tuning**: The core functionality of the repository is to fine-tune the Llama 2 model using your custom text data. The code is configured to allow easy adaptation for various text-based tasks.
- **Inference**: Once fine-tuned, the model can be used to generate text from a provided prompt.
- **Optimization**: The training script is optimized for performance, allowing for efficient training on both CPU and GPU.

## Frameworks and Libraries

### Hugging Face Transformers
- **Transformers** is a popular library for natural language processing (NLP) models, including pre-trained models like GPT, BERT, and Llama 2. This project uses it to load, fine-tune, and save the Llama 2 model.

### Hugging Face Datasets
- **Datasets** is a library that facilitates loading, processing, and managing large datasets. It provides easy-to-use interfaces for handling and streaming datasets, which is crucial for training large models like Llama 2.

### PyTorch
- **PyTorch** is an open-source deep learning framework that provides flexibility and performance. It is used as the backend for training the model and handling tensor operations.

### Accelerate
- **Accelerate** is a library by Hugging Face to speed up training on multi-GPU systems. It simplifies distributed training and model parallelism, making it easier to scale the training process.

### Other Libraries
- **numpy** and **pandas** are used for data manipulation and preparation.
- **torchmetrics** is used to calculate evaluation metrics during fine-tuning.

## Prerequisites

To run this project, ensure that you have the following installed:

- **Python 3.8+**: Python is required to run the scripts.
- **PyTorch** (with GPU support for better performance).
- **Hugging Face Transformers** and **Datasets**.
- **Accelerate** for distributed training.
- Other necessary libraries like `numpy`, `torchmetrics`, and `pandas`.

## Installation

Follow these steps to set up the project on your local machine.

### Step 1: Clone the Repository

Clone the repository to your local machine.

    ```bash
    git clone https://github.com/yourusername/llama-2-finetuning.git
    cd llama-2-finetuning

### Step 2: Install Required Dependencies

    ```bash
    pip install torch>=1.13.0
    pip install transformers>=4.30.0
    pip install datasets>=2.10.0
    pip install accelerate>=0.18.0
    pip install numpy>=1.21.0
    pip install torchmetrics>=0.11.0
    pip install pandas>=1.3.0

### Step 3: Setup the hf_token ( Optional : If you want to push the project to huggingface )

### Step 3: Execute the "Curating the data" Notebook to set up fine tuning

### Step 4: Execute the "Fine tune LLama" Notebook to fine tune and infer
