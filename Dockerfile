FROM python:3.9-slim

# Install Git LFS so Railway can pull large files
RUN apt-get update && apt-get install -y git-lfs && git lfs install

# Set working directory
WORKDIR /app

# Copy everything
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
