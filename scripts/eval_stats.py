import numpy as np
import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-npz", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    if not args.eval_npz.exists():
        print(f"Eval file {args.eval_npz} not found.")
        return

    data = np.load(args.eval_npz)
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
    with open(args.out_md, "w") as f:
        f.write(md_content)
    
    print(f"Stats written to {args.out_md}")

if __name__ == "__main__":
    main()
