from pipelines.training_pipeline import training_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
import click

# @click.option(
#     "--num-epochs",
#     default=1,
#     type=click.INT,
#     help="Number of epochs to train the model for.",
# )

def main():
    df = training_pipeline()
    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the experiment."
    )

if __name__ == "__main__":
    main()