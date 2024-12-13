from zenml import pipeline
from steps.huggingface_deploy.huggingface_deploy import upload_model_to_huggingface
from dotenv import load_dotenv
import sagemaker
import boto3
from pipelines.training_pipeline import training_pipeline
from sagemaker.huggingface import HuggingFaceModel
from steps.predictor.predictor import huggingface_predict, prediction_service_loader,predictor
load_dotenv()
import os 
from zenml import pipeline
from zenml.config import DockerSettings
from zenml.integrations.constants import HUGGINGFACE
from zenml.integrations.huggingface.services import HuggingFaceServiceConfig
from zenml.integrations.huggingface.steps import (
    huggingface_model_deployer_step,
)
import time
import boto3
from sagemaker.huggingface import HuggingFaceModel
from sagemaker import get_execution_role
from steps.predictor.test_data import get_text_data
token = os.getenv('HUGGINGFACE_TOKEN')


@pipeline(enable_cache=True)
def huggingface_uploader_pipeline():
    """
    A pipeline that trains a model and uploads it to the Hugging Face Model Hub.

    The pipeline consists of two steps: training and uploading. The training step
    trains a model using the `training_pipeline` pipeline. The uploading step
    uploads the trained model to the Hugging Face Model Hub using the
    `upload_model_to_huggingface` step.

    The pipeline takes no inputs and produces no outputs. The Hugging Face token
    should be set as an environment variable named HUGGINGFACE_TOKEN.

    Args:
        None

    Returns:
        None
    """
    model,tokenizer , model_path= training_pipeline()
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN")  # Replace with your token
    model_folder_path = model_path  # Path to your saved model folder
    repository_name = "new-model"  # Desired name for the repository

    upload_model_to_huggingface(repo_name=repository_name, 
                                model_folder=model_folder_path, 
                                token=huggingface_token)




docker_settings = DockerSettings(
    required_integrations=[HUGGINGFACE],
)

# It is an paid one for hugging service deployment , it charges for the inference prediction , so you can 
# skip this step.
@pipeline(enable_cache=True, settings={"docker": docker_settings})
def huggingface_deployment_pipeline(
    model_name: str = "hf",
    timeout: int = 1200,
):
    """
    Deploys a Hugging Face model to a specified cloud vendor.

    This pipeline configures and deploys a Hugging Face model to an endpoint
    using the provided service configuration. The deployment utilizes the
    specified model name, framework, region, and other configuration settings
    to set up the model on the chosen cloud infrastructure.

    Args:
        model_name (str, optional): The name of the model to deploy. Defaults to "hf".
        timeout (int, optional): The timeout for the deployment process in seconds. Defaults to 1200.

    Returns:
        None
    """

    service_config = HuggingFaceServiceConfig(model_name="distilbert-base-cased",
                                              endpoint_name="distilbert-base-cased-sara",
                                              repository="Sarathkumar1304ai/my-new-mode",
                                              framework='pytorch',
                                              instance_size="large",
                                              instance_type="g5.12xlarge",
                                              accelerator="cpu",
                                              region="us-east-1",
                                              vendor="aws",
                                              token=token,
                                              task = "text-classification",
                                              endpoint_type="public",
                                              namespace="sentiment_analysis")

    # Deployment step
    huggingface_model_deployer_step(
        service_config=service_config,
        timeout=timeout,
    )


# Deploy huggingface to aws sagemaker.
@pipeline
def huggingface_inference_pipeline():
    """
    Deploys a Hugging Face model to SageMaker Inference and performs inference with a test input.

    This pipeline deploys a Hugging Face model to SageMaker Inference and performs
    inference with a test input. The model is deployed with the specified
    execution role and instance configuration. The test input is a string that is
    passed to the model for inference.

    Args:
        None

    Returns:
        dict: The output of the model inference.
    """
    session = boto3.Session(region_name="us-east-1")  # Update with your region
    sagemaker = session.client('sagemaker')

    try:
        role = get_execution_role()
    except ValueError:
        iam = boto3.client('iam')
        role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']

    # Hub Model configuration
    hub = {
        'HF_MODEL_ID': 'Sarathkumar1304ai/my-pretrained-model',
        'HF_TASK': 'text-classification',
    }

    # Create Hugging Face Model
    huggingface_model = HuggingFaceModel(
        transformers_version='4.37.0',
        pytorch_version='2.1.0',
        py_version='py310',
        env=hub,
        role=role,
        sagemaker_session=session  # Use the session with the specified region
    )

    # Deploy model to SageMaker Inference
    predictor = huggingface_model.deploy(
        initial_instance_count=1,  # number of instances
        instance_type='ml.m5.xlarge'  # EC2 instance type
    )

    return predictor.predict({
        "inputs": "I like you. I love you",
    })




@pipeline(enable_cache=False)
def custom_inference_pipeline():
    """
    Performs inference with a custom input and displays the result.

    This pipeline calls the `huggingface_predict` step with a custom input
    and displays the result. The custom input is retrieved from the
    `get_text_data` function.

    Args:
        None

    Returns:
        dict: The output of the model inference.
    """
    time.sleep(20)
    result = huggingface_predict(input_data = get_text_data())
    print("results :",result)
    return result




@pipeline
def huggingface_deployment_inference_pipeline(
    pipeline_name: str="huggingface_deployment_pipeline", pipeline_step_name: str = "huggingface_model_deployer_step",
):
    """
    Runs inference on a custom input using a model deployed by the Hugging Face deployment pipeline.

    This pipeline runs inference on a custom input using a model deployed by the
    Hugging Face deployment pipeline. The custom input is retrieved from the
    `get_text_data` function. The model deployment service is retrieved from the
    `prediction_service_loader` step.

    Args:
        pipeline_name (str, optional): The name of the pipeline that deployed the model. Defaults to "huggingface_deployment_pipeline".
        pipeline_step_name (str, optional): The name of the step that deployed the model. Defaults to "huggingface_model_deployer_step".

    Returns:
        dict: The output of the model inference.
    """
    inference_data = get_text_data()
    model_deployment_service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
    )
    predictions = predictor(model_deployment_service, inference_data)
    return predictions