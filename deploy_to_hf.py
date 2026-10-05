"""
Upload the project to a HuggingFace Space (Docker SDK).
Usage:  python deploy_to_hf.py <hf-username>/eco-travel-advisor
Secrets (.env), trained models and conversation data are never uploaded.
"""
import sys
from huggingface_hub import HfApi

IGNORE = [".env", ".git/*", ".idea/*", ".rasa/*", "models/*", "models_split/*",
          "train_test_split/*", "handover_queue/*", ".pytest_cache/*",
          "**/__pycache__/*", "results/*"]

repo_id = sys.argv[1]
api = HfApi()
api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
api.upload_folder(folder_path=".", repo_id=repo_id, repo_type="space",
                  ignore_patterns=IGNORE, commit_message="Deploy Eco-Travel Advisor")
print(f"Uploaded. Watch the build at: https://huggingface.co/spaces/{repo_id}")
