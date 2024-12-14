from pipelines.deployment_pipeline import (huggingface_uploader_pipeline,huggingface_inference_pipeline,
 huggingface_deployment_pipeline, custom_inference_pipeline,huggingface_deployment_inference_pipeline)

def main():
    url = huggingface_uploader_pipeline()
    prediction = custom_inference_pipeline()
    print("prediction :",prediction)
    # huggingface_deployment_pipeline()
    # huggingface_inference_pipeline()
    # huggingface_deployment_inference_pipeline()
    return prediction
    
if __name__ == "__main__":
    main()



