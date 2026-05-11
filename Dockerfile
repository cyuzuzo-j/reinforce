FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (e.g., git for some python packages if needed)
RUN apt-get update && apt-get install -y git build-essential curl && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY pyproject.toml .
# We need to install the package itself and runpod
RUN pip install runpod
# Then copy the rest of the code and install it
COPY . .
RUN pip install -e .[train]

# Build sample data during image build to speed up cold starts
RUN python scripts/build_sample.py --top-n 50 --min-volume 5000 --out data/

# Run the RunPod handler
CMD ["python", "-u", "handler.py"]
