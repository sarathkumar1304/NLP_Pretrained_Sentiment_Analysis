from utils.logger import logger
from transformers import PreTrainedModel,PreTrainedTokenizer
from zenml import step, get_step_context
import os
from zenml.materializers.base_materializer import BaseMaterializer
@step(enable_cache=True)
def save_model_to_deploy(model:PreTrainedModel,tokenizer:PreTrainedTokenizer)->None:
    logger.info("Saving model")
    pipeline_extra = get_step_context().pipeline_run.config.extra
    logger.info(
        f"Loading latest version of the model for stage to deploy"
    )
    latest_version = get_step_context().model

    model = latest_version.load_artifact(name = "model")
    tokenizer = latest_version.load_artifact(name = "tokenizer")

    model_path = "./new_model/model"
    tokenizer_path = "./new_model/tokenizer"

    model.save_pretrained(model_path)
    tokenizer.save_pretrained(tokenizer_path)

    logger.info(
        f"model and tokenizer saved to {model_path} and {tokenizer_path}"
    )


