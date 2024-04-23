import subprocess

def run_commands():
    commands = [
        "export MLFLOW_TRACKING_URI=https://dagshub.com/kalema3502/MIIA-Pothole-Image-classification.mlflow",
        "export MLFLOW_TRACKING_USERNAME=kalema3502",
        "export MLFLOW_TRACKING_PASSWORD=fb3845efcc3b2e46a4157b1d2c977a21e02dd16e",
    ]
    
    for command in commands:
        subprocess.run(command, shell=True)

if __name__ == "__main__":
    run_commands()
