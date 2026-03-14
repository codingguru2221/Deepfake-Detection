# AWS Deployment Guide

This project has two deployable parts:

1. `FastAPI` backend for ML inference
2. `Deepfake-Frontend` React/Node frontend

## Recommended AWS path

For your current project, the easiest reliable path is:

1. Deploy backend on `EC2` or `ECS Fargate`
2. Deploy frontend on `Amplify` or as a small container on `App Runner`

If you want the fastest first launch, use `EC2` for backend and `Amplify` for frontend.

## Important before deploy

This repo ignores trained assets:

- `models/checkpoints/`
- `models/exports/`
- runtime data under `data/`

So before production, make sure your AWS server/container has:

1. trained model files
2. enough disk for uploads and runtime logs
3. enough RAM/CPU for TensorFlow and PyTorch inference

## Option A: Deploy with Docker to ECS/App Runner

### Backend image

Build from repo root:

```powershell
docker build -f Dockerfile.backend -t deepfake-backend .
```

Run locally:

```powershell
docker run -p 8000:8000 `
  -e PORT=8000 `
  -e DF_CRAWLER_ENABLED=0 `
  -e DF_RUNTIME_LEARNING_ENABLED=0 `
  deepfake-backend
```

### Frontend image

Build from repo root:

```powershell
docker build -f Deepfake-Frontend/Dockerfile `
  --build-arg VITE_API_BASE_URL=http://YOUR_BACKEND_URL:8000 `
  -t deepfake-frontend ./Deepfake-Frontend
```

Run locally:

```powershell
docker run -p 5000:5000 -e PORT=5000 deepfake-frontend
```

### Local production smoke test

```powershell
docker compose -f docker-compose.prod.yml up --build
```

## Option B: Backend on EC2

This is the easiest if your model files are large and you want full control.

### EC2 setup

1. Launch an Ubuntu EC2 instance
2. Open inbound ports:
   - `22` for SSH
   - `8000` for FastAPI, or keep it private behind Nginx
   - `80` and `443` for public traffic
3. Install Docker
4. Copy project and model files to the instance
5. Start backend container

Example:

```bash
docker build -f Dockerfile.backend -t deepfake-backend .
docker run -d \
  --name deepfake-backend \
  -p 8000:8000 \
  -e PORT=8000 \
  -e DF_CRAWLER_ENABLED=0 \
  -e DF_RUNTIME_LEARNING_ENABLED=0 \
  deepfake-backend
```

Then put `Nginx` in front and proxy `/` or `/api` to port `8000`.

## Option C: Frontend on Amplify

If you only need the React UI publicly hosted, Amplify is simplest.

Use build settings similar to:

```yaml
version: 1
frontend:
  phases:
    preBuild:
      commands:
        - cd Deepfake-Frontend
        - npm ci
    build:
      commands:
        - npm run build
  artifacts:
    baseDirectory: Deepfake-Frontend/dist/public
    files:
      - "**/*"
  cache:
    paths:
      - Deepfake-Frontend/node_modules/**/*
```

Set env var:

- `VITE_API_BASE_URL=https://your-backend-domain`

## Production env vars

Backend commonly needs:

- `PORT=8000`
- `PYTHONPATH=/app/src`
- `DF_IMAGE_MODEL=/app/models/exports/image_tf_model.keras`
- `DF_VIDEO_MODEL=/app/models/checkpoints/video_gru.pt`
- `DF_AUDIO_MODEL=/app/models/exports/audio_rf.joblib`
- `DF_CRAWLER_ENABLED=0`
- `DF_RUNTIME_LEARNING_ENABLED=0`

Optional AWS Rekognition Custom Labels path for hackathon demos:

- `AWS_REKOGNITION_ENABLED=1`
- `AWS_REGION=us-east-1`
- `AWS_REKOGNITION_PROJECT_VERSION_ARN=arn:aws:rekognition:...`
- `AWS_REKOGNITION_MIN_CONFIDENCE=50`
- `AWS_REKOGNITION_VIDEO_FRAMES=8`

If enabled, image and video inference can use AWS-hosted custom labels instead of local image/video models.

Optional Hugging Face pretrained path for faster hackathon demos without retraining:

- `HF_DEEPFAKE_ENABLED=1`
- `HF_DEEPFAKE_MODEL_ID=prithivMLmods/deepfake-detector-model-v1`
- `HF_DEEPFAKE_VIDEO_FRAMES=8`

If enabled, image inference uses the Hugging Face SigLIP deepfake detector directly, and video inference runs frame sampling through the same image detector.

Frontend commonly needs:

- `PORT=5000`
- `VITE_API_BASE_URL=https://your-backend-domain`

## My recommendation for this repo

Start with:

1. `Backend`: EC2 using `Dockerfile.backend`
2. `Frontend`: Amplify using `Deepfake-Frontend`

That keeps deployment simple while your ML side stays easy to debug.
