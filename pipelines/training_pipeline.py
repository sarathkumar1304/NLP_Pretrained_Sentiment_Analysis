from steps.data_loader.data_ingestion import data_ingestion
from steps.tokenization.tokenization import tokenization_step
from steps.tokenizer_loader.tokenizer_loader import tokenizer_loader
from steps.training.model_trainer import model_trainer
from zenml import pipeline,Model,step
from transformers import DistilBertForSequenceClassification,DistilBertTokenizer
from typing import Optional
from steps.deploy.save_model import prepare_model_step
from zenml import get_pipeline_context
from utils.logger import logger
import click

# @click


@pipeline(enable_cache=True,
          model = Model(name = "text_classification"))
def training_pipeline(
    lower_case:bool=True,
    padding:Optional[bool] =True,
    max_length:Optional[int] =512,
    text_column:Optional[str] = 'text',
    label_column:Optional[str] = 'label',
    train_batch_size:Optional[int] = 4,
    eval_batch_size:Optional[int] = 4,
    num_epochs:Optional[int] = 3,
    learning_rate:Optional[float] = 2e-5,
    weight_decay:Optional[float] = 0.01,
):
    """
    A pipeline for training a text classification model using DistilBERT.

    Args:
        lower_case (bool): Whether to convert text to lowercase during tokenization.
        padding (Optional[bool]): Whether to pad sequences during tokenization.
        max_length (Optional[int]): Maximum sequence length for tokenization.
        text_column (Optional[str]): The column name containing text data.
        label_column (Optional[str]): The column name containing label data.
        train_batch_size (Optional[int]): Batch size for training.
        eval_batch_size (Optional[int]): Batch size for evaluation.
        num_epochs (Optional[int]): Number of epochs for training.
        learning_rate (Optional[float]): Learning rate for the optimizer.
        weight_decay (Optional[float]): Weight decay for the optimizer.

    Returns:
        tuple: Contains the trained model, tokenizer, and the path to the saved model.
    """

    dataset = data_ingestion()
    tokenizer= tokenizer_loader(lower_case=lower_case)
    tokenized_dataset = tokenization_step(
        dataset=dataset,
        tokenizer=tokenizer,
        padding=padding,
        max_length=max_length,
        text_column=text_column,
        label_column=label_column,
    )
    model,tokenizer,model_path = model_trainer(
        tokenized_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    return model,tokenizer, model_path
    

# zenml model-deployer register huggingface_model_deployer --flavor=huggingface --token=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 --namespace=text_classification
# zenml stack update sentiment_analysis_stack --model-deployer=huggingface_model_deployer_new