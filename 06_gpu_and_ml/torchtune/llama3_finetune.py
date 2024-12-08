import subprocess
from pathlib import Path

import modal

REMOTE_CONFIG_PATH = Path("/llama3_3_70B_full.yaml")
REMOTE_OUTPUT_DIR = Path("/data/torchtune/llama3_3_70B/full")

app = modal.App(name="llama3-finetune")

volume = modal.Volume.from_name(
    "llama3-finetune-volume", create_if_missing=True
)

image = (
    modal.Image.debian_slim()
    .pip_install("wandb", "torch", "torchao", "torchvision")
    .apt_install("git")
    .pip_install(
        "git+https://github.com/pytorch/torchtune.git@06a837953a89cdb805c7538ff5e0cc86c7ab44d9"
    )
    .add_local_file("llama3_3_70B_full.yaml", REMOTE_CONFIG_PATH.as_posix())
)


@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=[
        modal.Secret.from_name("huggingface-secret", environment_name="main")
    ],
    timeout=60 * 60,
)
def download_model():
    REMOTE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "tune",
            "download",
            "unsloth/Llama-3.3-70B-Instruct",
            "--output-dir",
            REMOTE_OUTPUT_DIR.as_posix(),
            "--ignore-patterns",
            "original/consolidated.00.pth",
        ]
    )


@app.function(
    image=image, gpu="H100:8", volumes={"/data": volume}, timeout=60 * 60 * 24
)
def finetune():
    subprocess.run(
        [
            "tune",
            "run",
            "--nproc_per_node",
            "8",
            "full_finetune_distributed",
            "--config",
            REMOTE_CONFIG_PATH,
        ]
    )
