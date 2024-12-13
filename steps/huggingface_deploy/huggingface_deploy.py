from huggingface_hub import HfApi, HfFolder, Repository
import os
from utils.logger import logger
# import logging
from zenml import step
import os 


readme_content = """
---
tags:
  - sentiment-analysis
  - text-classification
  - transformers
  - bert
  - zenml
  - mlops
license: mit
pipeline_tag: text-classification
language:
  - en
datasets:
  - stanfordnlp/imdb
base_model:
  - distilbert/distilbert-base-cased
framework:
  - pytorch
  - huggingface
platforms:
  - streamlit
  - docker
  - zenml
---

### ZenML Sentiment Analysis Demo 🎥📊

This is a demo model showcasing the integration of **ZenML** for MLOps pipeline automation in a **Sentiment Analysis** project. The project uses **Hugging Face Transformers** for text classification on the **IMDb Movie Reviews Dataset**, fine-tuning the `distilbert-base-uncased` model to predict whether reviews are **positive** or **negative**.

#### Key Features of the Project:
- **Pre-trained Transformers**: Leverages Hugging Face’s `distilbert-base-cased` for state-of-the-art text classification.
- **ZenML Pipelines**: Automated workflows for data ingestion, model training, evaluation, and deployment.
- **Streamlit Interface**: A user-friendly web app for real-time predictions.
- **MLOps Best Practices**: MLflow for experiment tracking and Docker for containerized deployment.
- **Scalable Deployment**: Easy to extend and reproduce for real-world applications.

---

#### 🛠️ Tools and Technologies
- **ZenML**: Simplifies MLOps workflows for reproducibility.
- **Hugging Face Transformers**: Provides pre-trained models for fine-tuning.
- **PyTorch**: Framework for deep learning.
- **MLflow**: Experiment tracking and model versioning.
- **Streamlit**: Interactive frontend for user predictions.
- **Docker**: Containerization for deployment.
- **IMDb Dataset**: Dataset of 50,000 labeled reviews for training and evaluation.

---

#### 📈 Model Overview
- **Model Base**: `distilbert-base-uncased`
- **Dataset**: [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb)
- **Metrics**:
  - Accuracy: 97% (add value after testing)
  - Precision: 1.0%
  - Recall: 1.0%
  - F1-Score: 1.0%
- **Usage**:
  - Input: Movie review text (e.g., "This movie was fantastic!")
  - Output: Sentiment (Positive/Negative) with confidence scores.

---

#### 🚀 How to Use
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/sarathkumar1304/NLP-Pretrained-sentiment-analysis.git
   cd NLP-Pretrained-sentiment-analysis


"""



@step(enable_cache=True)
def upload_model_to_huggingface(repo_name:str, model_folder:str, token:str)->str:
    """
    Uploads a pretrained model to the Hugging Face Hub.

    Args:
        repo_name (str): The name of the repository to create on Hugging Face.
        model_folder (str): The path to the folder containing the pretrained model.
        token (str): Your Hugging Face authentication token.

    Returns:
        str: URL of the uploaded repository.
    """
    try:
        logger.info("Uploading model to Hugging Face Hub")
        # Set up Hugging Face credentials
        HfFolder.save_token(token)
        api = HfApi()

        # Check if the repository exists, create if not
        username = api.whoami(token=token)["name"]
        repo_id =  f"{username}/{repo_name}"
        repo_url = f"https://huggingface.co/{repo_id}"
        if not api.repo_exists(repo_id=repo_id, token=token):
            logger.info("Creating repository on Hugging Face Hub")
            api.create_repo(repo_id=repo_id, token=token)
            logger.info("Repository created on Hugging Face Hub")

        # Clone or initialize the repository
        
        logger.info("Cloning or initializing repository")
        repo = Repository(local_dir=repo_name, clone_from=repo_url, token=token)

        logger.info("Adding model folder to repository")
        
        logger.info("Adding README.md file to repository")
        readme_path = os.path.join(repo.local_dir, "README.md")
        with open(readme_path, "w") as readme_file:
            readme_file.write(readme_content)

        # Copy the model files to the repository folder
        for filename in os.listdir(model_folder):
            src = os.path.join(model_folder, filename)
            dest = os.path.join(repo.local_dir, filename)
            if os.path.isfile(src):
                os.replace(src, dest)

        # Push changes to the Hugging Face Hub
        repo.git_add()
        repo.git_commit("Initial commit of pretrained model")
        repo.git_push()

        print(f"Model uploaded successfully: {repo_url}")
        logger.info(f"Model uploaded successfully: {repo_url}")
        # logging.info("Model uploaded successfully : %s", repo_url)
       
        return repo_url

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Call the function
if __name__ == "__main__":
    huggingface_token = "your_huggingface_token"  # Replace with your token
    model_folder_path = "your_saved_model"  # Path to your saved model folder
    repository_name = "my-pretrained-model"  # Desired name for the repository

    upload_model_to_huggingface(repo_name=repository_name, 
                                model_folder=model_folder_path, 
                                token=huggingface_token)
