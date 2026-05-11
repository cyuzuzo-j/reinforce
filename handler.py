import os
import subprocess
import runpod
from github import Github
import numpy as np

def run_training(total_timesteps):
    out_dir = "runs/runpod_train"
    # Make sure sample data exists
    if not os.path.exists("data/markets.parquet"):
        subprocess.run(["python", "scripts/build_sample.py", "--top-n", "50", "--min-volume", "5000", "--out", "data/"], check=True)
    
    cmd = [
        "python", "scripts/train.py",
        "--data-dir", "data/",
        "--total-timesteps", str(total_timesteps),
        "--n-envs", "4",
        "--out-dir", out_dir
    ]
    subprocess.run(cmd, check=True)
    return out_dir

def get_stats_markdown(out_dir):
    eval_npz = os.path.join(out_dir, "eval_log", "evaluations.npz")
    if not os.path.exists(eval_npz):
        return "No evaluation stats available."
        
    data = np.load(eval_npz)
    results = data["results"]
    ep_lengths = data["ep_lengths"]
    timesteps = data["timesteps"]

    last_eval_rewards = results[-1]
    last_eval_lengths = ep_lengths[-1]

    mean_reward = np.mean(last_eval_rewards)
    std_reward = np.std(last_eval_rewards)
    mean_length = np.mean(last_eval_lengths)

    md_content = f"""## Training Evaluation Statistics

* **Timesteps trained:** {timesteps[-1]}
* **Mean Reward:** {mean_reward:.2f} ± {std_reward:.2f}
* **Mean Episode Length:** {mean_length:.2f}
"""
    return md_content

def update_github(token, repo_name, pr_number, release_id, stats_md):
    g = Github(token)
    repo = g.get_repo(repo_name)
    
    messages = []
    
    if pr_number:
        pr = repo.get_pull(pr_number)
        pr.create_issue_comment(stats_md)
        messages.append(f"Successfully posted comment on PR #{pr_number}")
        
    if release_id:
        try:
            # First try by id if it's numeric
            release = repo.get_release(int(release_id))
        except ValueError:
            # Fallback to tag name
            release = repo.get_release(release_id)
            
        new_body = (release.body or "") + "\n\n" + stats_md
        release.update_release(release.title, new_body)
        messages.append(f"Successfully updated release {release_id}")
        
    return messages

def handler(job):
    job_input = job.get("input", {})
    
    total_timesteps = job_input.get("total_timesteps", 10000)
    github_token = job_input.get("github_token", os.environ.get("GITHUB_TOKEN", ""))
    github_repo = job_input.get("github_repo", "cyuzuzo-j/reinforce")
    release_id = job_input.get("release_id", "")
    pr_number = job_input.get("pr_number", 0)
    
    print(f"Starting training with {total_timesteps} timesteps...")
    try:
        out_dir = run_training(total_timesteps)
        stats_md = get_stats_markdown(out_dir)
        print("Training complete. Stats:\n", stats_md)
        
        result = {
            "status": "success",
            "stats": stats_md,
            "messages": []
        }
        
        if github_token and (pr_number or release_id):
            print("Updating GitHub...")
            msgs = update_github(github_token, github_repo, pr_number, release_id, stats_md)
            result["messages"] = msgs
            print("GitHub updated:", msgs)
        else:
            print("No GitHub token or target provided, skipping GitHub update.")
            
        return result
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
