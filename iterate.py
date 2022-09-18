# iterate.py 


import click
import mlflow
from pathlib import Path
import pickle
from mlflow.tracking import MlflowClient

@click.command()
@click.option(
    "--index",
    "-i",
    type=click.INT,
    help="Upper limit for iteration",
)
@click.option(
    "--log-dir",
    "-L",
    type=click.Path(),
    default=Path("log_artifacts"),
    help="Log artifact file path"
)
@click.option(
    "--model-dir",
    "-M",
    type=click.Path(),
    default=Path("model_dir"),
    help="'Model' directory"
)
def iterate(index: int, log_dir: Path, model_dir: Path) -> None:
    if log_dir is None or str(log_dir).strip() == "": 
        cfg_log_dir = Path("log_artifacts")
    else:
        cfg_log_dir = Path(log_dir)

    if model_dir is None or str(model_dir).strip() == "": 
        cfg_model_dir = Path("model_dir")
    else:
        cfg_model_dir = Path(model_dir)
    
    if not cfg_log_dir.exists(): cfg_log_dir.mkdir()
    if not cfg_model_dir.exists(): cfg_model_dir.mkdir()

    artifact_file = cfg_log_dir / "log_file.txt"

    with mlflow.start_run() as run:
        # Fetch run information
        run_id = run.info.run_id
        run = MlflowClient().get_run(run_id=run_id)
        logged_params = run.data.params

        # Set param dict
        params_dict = {
            "index": index,
            "log_dir": str(log_dir),
            "model_dir": str(model_dir),
            "cfg_log_dir": str(cfg_log_dir),
            "cfg_model_dir": str(cfg_model_dir)
        }

        # Log if not already logged
        for k, v in params_dict.items():
            if k not in logged_params:
                mlflow.log_param(key=k, value=str(v))

        # iterate and log
        iter_and_log(index, artifact_file)

        # log cfg_log_dir as artifact
        mlflow.log_artifacts(str(cfg_log_dir), "log_dir")

        # binarize and save model
        with open(cfg_model_dir / "model_pickle", "wb") as model_file:
            pickle.dump(iter_and_log, model_file)

        # log as mlflow model that can be registered
        mlflow.pyfunc.log_model(
            artifact_path="iterate_model", 
            artifacts={
                "model_dir": str(cfg_model_dir)
            },
            python_model=mlflow.pyfunc.PythonModel()
        )


def iter_and_log(index: int, log_file: Path):
    for i in range(index):
        mlflow.log_metric(key="current_iter", value=i, step=i)
        with open(log_file, "w") as lf:
            lf.write(f"{i}\n")
    

if __name__ == "__main__":
    iterate()