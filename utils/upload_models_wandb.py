import wandb
import argparse


if __name__ == "__main__":
    project = "backdoor-attack"
    entity = "nguyenhongsonk62hust"
    wandb_path = "../hdd/home/ssd_data/Son/Venomancer/wandb/wandb"

    parser = argparse.ArgumentParser(description='Uploading models to wandb')
    parser.add_argument('--run_id', dest='run_id', help="wandb run id", required=True)
    parser.add_argument('--file_path', dest='file_path', help="Path to the model file that needs to upload",required=True)
    parser.add_argument('--exp_name', dest='exp_name', help='Experiment name', required=True)
    args = parser.parse_args()
    run_id = args.run_id
    file_path = args.file_path
    exp_name = args.exp_name

    with wandb.init(project=project, entity=entity, dir=wandb_path, id=run_id, resume="allow") as run:
        model_name = file_path.split("/")[-1]
        artifact = wandb.Artifact(model_name, exp_name)
        artifact.add_file(file_path)
        run.log_artifact(artifact)

    # Example usage:
    # python upload_models_wandb.py --run_id <run id> --file_path <file path> --exp_name <experiment name>
    