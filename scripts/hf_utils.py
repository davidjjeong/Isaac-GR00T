import os
from huggingface_hub import snapshot_download, HfApi

def download_data(repo_id: str = "IPEC-COMMUNITY/libero_90_no_noops_lerobot", local_dir: str = "./demo_data"):
    folder_path = snapshot_download(
        repo_id = repo_id,
        repo_type = "dataset",
        local_dir = local_dir,
        token = os.environ.get("HF_TOKEN")
    )

def upload_checkpoint(
        repo_id: str = "davidjjeong/gr00t_finetuned_libero_90_v0",
        checkpoint_path: str = "checkpoints/gr00t_libero",
        private: bool = True,
):
    api = HfApi()

    # Create a new repo on Hugging Face
    api.create_repo(
        repo_id=repo_id,
        token=os.environ.get("HF_TOKEN"),
        private=private,
        repo_type="model",
        exist_ok=True
    )

    # Upload large files (safetensors)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=checkpoint_path,
        token=os.environ.get("HF_TOKEN"),
        repo_type="model",
        allow_patterns=["*.safetensors"],
        ignore_patterns=["*/*"]
    )