# CS159-Caltech
CS159 2024 LLM Project:  CrossAttentionDTI: A Synergistic Approach to Drug-Target Interaction Prediction with Pretrained Protein Language Model ESM1b, and Llama-3 LLM

This repository contains two main files designed for different aspects of drug-target interaction (DTI) modeling and analysis:

CS159 DTI Training.ipynb: This notebook is dedicated to obtaining protein embeddings, fine-tuning the ESM1b model, and training the CrossAttentionDTI model using a cross-attention mechanism.

CS159 DTI AI Agent.ipynb: This notebook leverages the Llama 3 8B-Instruct model to access biochemical contextual information about drugs. It includes three demonstration examples to showcase its capabilities.


Important Notes:

1. Both notebooks were created and tested on Google Colab using an L4 GPU for training.
2. All necessary installations and dependencies are included within the notebooks, ensuring they can run successfully on Colab.
3. You should create your own HuggingFace account for accessing the ESM1b model.
4. You should also provide your own URL for accessing the Llama 3 model.
