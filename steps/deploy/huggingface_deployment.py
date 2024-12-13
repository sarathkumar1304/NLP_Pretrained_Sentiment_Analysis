from typing import Optional, Annotated
from sagemaker.huggingface import HuggingFaceModel
from zenml import get_step_context,step
from utils.logger import logger
from gradio.aws_helper import get_sagemaker_role,get_sagemaker_session

def deploy_hf_to_sagemaker(
        repo_id:Optional[str]= None,
        revision :Optional[str]= None,
        transformers_version:str="4.46.3",
        pytorch_version:str = "2.5.1+cpu",
        py_version:str = 'py312',
        hf_task :str = "text-classification",
        instance_type :str = "ml.g5.2xlarge",
        container_startup_health_check_timeout:int= 300,
)->Annotated['str',"sagemaker_endpoint_name"]:
    """
    Deploys a HuggingFace model to SageMaker.

    This function takes in the id and revision of the model to deploy, as well as
    the transformers version, pytorch version, and python version to use. It
    uses the `HuggingFaceModel` class from the SageMaker SDK to deploy the model
    to SageMaker.

    If the `repo_id` and `revision` arguments are not provided, the function
    will look for them in the `huggingface_url` artifact of the current
    pipeline step.

    Args:
        repo_id (str, optional): The id of the model to deploy. Defaults to None.
        revision (str, optional): The revision of the model to deploy. Defaults to None.
        transformers_version (str, optional): The transformers version to use. Defaults to "4.46.3".
        pytorch_version (str, optional): The pytorch version to use. Defaults to "2.5.1+cpu".
        py_version (str, optional): The python version to use. Defaults to 'py312'.
        hf_task (str, optional): The HuggingFace task to use. Defaults to "text-classification".
        instance_type (str, optional): The instance type to use. Defaults to "ml.g5.2xlarge".
        container_startup_health_check_timeout (int, optional): The timeout to use for the
            container startup health check. Defaults to 300.

    Returns:
        str: The name of the SageMaker endpoint created.
    """
    if repo_id is None or revision is None:
        context = get_step_context()
        mv = context.model_version
        deployment_metadata = mv.get_data_artifact(name="huggingface_url").run_metadata
        repo_id = deployment_metadata["repo_id"].values
        revision = deployment_metadata['revision'].values
    
    role = get_sagemaker_role()
    session = get_sagemaker_session()

    hub = {
        "HF_MODEL_ID":repo_id,
        "HF_MODEL_REVISION":revision,
        "HF_TASK":hf_task,
    }

    huggingface_model = HuggingFaceModel(
        env=hub,
        role=role,
        transformers_version=transformers_version,
        pytorch_version=pytorch_version,
        py_version=py_version,
        sagemaker_session = session
    )

    predictor = huggingface_model.deploy(
        initial_instance_count=1,
        instance_type = instance_type,
        container_startup_health_check_timeout=container_startup_health_check_timeout
    )

    endpoint_name = predictor.endpoint_name
    logger.info(f"Model deployed to sagemaker : {endpoint_name}")
    return endpoint_name
