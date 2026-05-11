import argparse
import asyncio
import platform

import torch


from runpod_flash import DataCenter, Endpoint, GpuType, NetworkVolume

training_vol = NetworkVolume(
    name="trading-vol-us",
    size=20,
    datacenter=DataCenter.US_KS_2,
)


@Endpoint(
    name="polymarket-gym-train",
    gpu=GpuType.ANY,
    dependencies=["torch", "duckdb", "gymnasium", "pandas","huggingface_hub" , "stable_baselines3"],
    volume=training_vol,
)
async def train(total_timesteps: int = 10000) -> dict:
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "no GPU"

    return {
        "status": "success",
        "total_timesteps": total_timesteps,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=10000)
    args = parser.parse_args()

    result = await train(args.total_timesteps)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
