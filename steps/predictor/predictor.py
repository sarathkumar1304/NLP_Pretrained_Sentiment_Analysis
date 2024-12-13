from rich import print
import requests
import os
import json
from dotenv import load_dotenv
from zenml import step
import time
load_dotenv()

# Define API details
API_URL = "https://api-inference.huggingface.co/models/Sarathkumar1304ai/my-pretrained-model"
API_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
headers = {"Authorization": f"Bearer {API_TOKEN}"} 
@step(enable_cache=False)
def huggingface_predict(input_data:str)->list:
    # Data for prediction
    
    """
    Sends a request to the Hugging Face API to perform sentiment analysis on the input data.

    Args:
        input_data (str): The input data to be analyzed.

    Returns:
        list: A list containing the sentiment analysis output.

    Raises:
        Exception: If the API request fails.
    """
    data = {"inputs":input_data } 
    time.sleep(20) # Loading the model take time by default 20 s 
    # Send the request
    response = requests.post(API_URL, headers=headers, json=data)  

    # Check response status
    if response.status_code == 200:
        output = response.json()  
        print(output[0])
        return output
    else:
        print(f"Error: {response.status_code}, {response.text}")

from typing import Annotated
from zenml import step, pipeline
from zenml.integrations.huggingface.model_deployers import HuggingFaceModelDeployer
from zenml.integrations.huggingface.services import HuggingFaceDeploymentService


# Load a prediction service deployed in another pipeline
@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "default",
) -> HuggingFaceDeploymentService:
    """Get the prediction service started by the deployment pipeline.

    Args:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """
    # get the Hugging Face model deployer stack component
    model_deployer = HuggingFaceModelDeployer.get_active_model_deployer()

    # fetch existing services with same pipeline name, step name and model name
    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )

    if not existing_services:
        raise RuntimeError(
            f"No Hugging Face inference endpoint deployed by step "
            f"'{pipeline_step_name}' in pipeline '{pipeline_name}' with name "
            f"'{model_name}' is currently running."
        )

    return existing_services[0]


# Use the service for inference
@step
def predictor(
    service: HuggingFaceDeploymentService,
    data: str
) -> Annotated[str, "predictions"]:
    """Run a inference request against a prediction service"""

    prediction = service.predict(data)
    return prediction



