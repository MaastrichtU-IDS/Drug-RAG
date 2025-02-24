# Drug Efficacy Justification Using Retrieval-Augmented Generation (RAG)

## Overview  
This repository contains the implementation of a **Retrieval-Augmented Generation (RAG) pipeline** designed to provide scientifically grounded justifications for drug efficacy claims. By leveraging **Large Language Models (LLMs)** and integrating verified biomedical data sources such as **PubMed** and **DrugBank**, this project aims to enhance the reliability of AI-driven justifications in biomedical research.  

## Features  
- **RAG-based Justification**: Retrieves task-specific, verified information to support drug-disease relationships.  
- **Multiple LLM Testing**: Evaluates different LLMs with various RAG techniques to determine the optimal model.  
- **Expert-Guided Evaluation**: Utilizes expert-curated ground truth for performance assessment.  
- **Role-Play Reasoning**: Employs scenario-based reasoning to enhance the logical consistency of generated justifications.  

## Installation  

### Prerequisites  
- Python 3.8+  
- CUDA-enabled GPU (optional but recommended)  
- Dependencies listed in `requirements.txt`  

### Clone the Repository  
```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
