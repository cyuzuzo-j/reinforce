import os
import subprocess
import runpod
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

def handler(job):
    job_input = job.get("input", {})
    total_timesteps = job_input.get("total_timesteps", 10000)
    
    print(f"Starting training with {total_timesteps} timesteps...")
    try:
        out_dir = run_training(total_timesteps)
        stats_md = get_stats_markdown(out_dir)
        print("Training complete. Stats:\n", stats_md)
        
        return {
            "status": "success",
            "stats": stats_md
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
