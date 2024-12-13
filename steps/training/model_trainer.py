import os
from transformers import( DistilBertForSequenceClassification, 
                         PreTrainedTokenizerBase,PreTrainedModel,TrainingArguments,Trainer)
from zenml import step, ArtifactConfig
from zenml.client import Client
from typing import Annotated, Tuple, Optional
from datasets import DatasetDict
import mlflow
from zenml import log_metadata
from steps.training.compute_metrics import compute_metrics
from utils.logger import logger
from transformers import DataCollatorWithPadding
import mlflow.transformers

experiment_tracker = Client().active_stack.experiment_tracker

@step(enable_cache=False, experiment_tracker=experiment_tracker.name)
def model_trainer(
    tokenized_dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    num_labels: int = 2,
    train_batch_size: Optional[int] = 16,
    num_epochs: Optional[int] = 3,
    learning_rate: Optional[float] = 2e-5,
    load_best_model_at_end: Optional[bool] = True,
    eval_batch_size: Optional[int] = 16,
    weight_decay: Optional[float] = 0.01,
    mlflow_model_name: Optional[str] = "sentiment_analysis",
) -> Tuple[
    Annotated[PreTrainedModel, ArtifactConfig(name="model", is_model_artifact=True)],
    Annotated[PreTrainedTokenizerBase, ArtifactConfig(name="tokenizer", is_model_artifact=True)],
    str,  # Path to the model directory
]:
    # Access train and validation splits dynamically
    """
    Train a DistilBERT model on the given dataset.

    Args:
    tokenized_dataset: A dictionary containing the tokenized dataset.
    tokenizer: A PreTrainedTokenizerBase instance.
    num_labels: The number of labels for the classification task.
    train_batch_size: The batch size for training.
    num_epochs: The number of epochs to train the model.
    learning_rate: The learning rate for training.
    load_best_model_at_end: Whether to load the best model at the end of the training.
    eval_batch_size: The batch size for evaluation.
    weight_decay: The weight decay for training.
    mlflow_model_name: The name for the model in MLflow.

    Returns:
    A tuple containing the trained model, the tokenizer, and the path to the model directory.
    """
    train_dataset = tokenized_dataset["train"]
    if "validation" not in tokenized_dataset:
        logger.info("Creating validation split from training data")
        # Split the train dataset into train and validation (90% train, 10% validation)
        train_dataset, eval_dataset = train_dataset.train_test_split(test_size=0.2).values()
        tokenized_dataset["validation"] = eval_dataset
    else:
        eval_dataset = tokenized_dataset["validation"]

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Start a new MLflow run
    if mlflow.active_run():
        logger.warning("Ending the previous active run.")
        mlflow.end_run()

    with mlflow.start_run(log_system_metrics=True) as run:
        logger.info(f"MLflow Run ID: {run.info.run_id}")
        logger.info("Model training started")
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-cased", num_labels=num_labels)
        
        training_args = TrainingArguments(
            output_dir="zenml_artifact",
            learning_rate=learning_rate,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            num_train_epochs=num_epochs,
            weight_decay=weight_decay,
            eval_strategy="steps",
            save_strategy="steps",
            save_steps=1000,
            eval_steps=100,
            logging_steps=10,
            save_total_limit=5,
            report_to="mlflow",
            load_best_model_at_end=load_best_model_at_end,
        )

        mlflow.transformers.autolog()

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
        )
        trainer.train()
        eval_results = trainer.evaluate(metric_key_prefix="")

        log_metadata(metadata={"metrics": eval_results}, artifact_name="model")

        logger.info("Model training completed")
        eval_metrics = trainer.evaluate(metric_key_prefix="")
        logger.info(eval_metrics)

        log_metadata(metadata={"metrics": eval_metrics}, artifact_name="model")

        components = {
            "model": model,
            "tokenizer": tokenizer
        }

        # Log model to MLflow
        mlflow.transformers.log_model(
            transformers_model=components,
            artifact_path=mlflow_model_name,
            register_model_name=mlflow_model_name,
            task='text-classification'
        )

        # Save model and tokenizer locally
        model_path = "./new_model"
        
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

        # Return model, tokenizer, and paths
        return model, tokenizer, model_path



