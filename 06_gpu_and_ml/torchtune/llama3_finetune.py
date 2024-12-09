# Usage
# ```bash
# modal run llama3_finetune.py::download_model
# ```
# then
# ```bash
# modal run llama3_finetune.py::finetune
# ```

import subprocess
from pathlib import Path

import modal

REMOTE_CONFIG_PATH = Path("/llama3_3_70B_full.yaml")
REMOTE_OUTPUT_DIR = Path("/data/torchtune/llama3_3_70B/full")
here = Path(__file__).parent

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
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_file(
        here / "llama3_3_70B_full.yaml", REMOTE_CONFIG_PATH.as_posix()
    )
)


@app.function(
    image=image,
    volumes={"/data": volume},
    # secrets=[  # only needed if downloading gated models
    #     modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])
    # ],
    timeout=60 * 60,  # set defensively high, should finish in ~10 minutes
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
def finetune(cli_overrides: str = None):
    # You can provide all the CLI arguments you wish to override
    # as a single string when calling this function, like
    #
    # ```bash
    # modal run llama3_finetune.py::finetune --cli-overrides "tokenizer.path=/home/my_tokenizer_path"
    # ```
    #
    # For more info on providing overrides, see the torchtune docs:
    # https://pytorch.org/torchtune/stable/deep_dives/configs.html#cli-override
    import shlex

    if cli_overrides is not None:
        cli_overrides = shlex.split(cli_overrides)
    else:
        cli_overrides = []

    # run a quick validation check
    subprocess.run(["tune", "validate", "--config", REMOTE_CONFIG_PATH])

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
        + cli_overrides
    )
