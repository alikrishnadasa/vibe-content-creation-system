# RunPod Deployment Guide

## Prerequisites

1. **Docker Hub Account**: Create account at https://hub.docker.com
2. **RunPod Account**: Sign up at https://www.runpod.io
3. **Local Docker**: Install Docker Desktop

## Step 1: Build and Push Docker Image

```bash
# Navigate to project directory
cd unified-video-system-main

# Build for AMD64 (RunPod architecture)
docker build --platform linux/amd64 -t your-dockerhub-username/unified-video-system:latest .

# Push to Docker Hub
docker push your-dockerhub-username/unified-video-system:latest
```

## Step 2: Create RunPod Template

1. Go to RunPod Dashboard → Serverless → Templates
2. Click "New Template"
3. Use the configuration from `runpod_template.json`:
   - **Name**: Unified Video System
   - **Docker Image**: `your-dockerhub-username/unified-video-system:latest`
   - **Container Disk**: 20 GB
   - **Volume**: 50 GB (for MJAnime clips and output)

## Step 3: Configure Environment Variables

Set these environment variables in your RunPod template:

```
CLIPS_DIRECTORY=/app/data/MJAnime
METADATA_FILE=/app/data/MJAnime/metadata_final_clean_shots.json
SCRIPTS_DIRECTORY=/app/data/scripts
MUSIC_FILE=/app/music/Beanie (Slowed).mp3
OUTPUT_DIRECTORY=/app/output
PYTHONPATH=/app
CUDA_VISIBLE_DEVICES=0
```

## Step 4: Upload Data to RunPod Volume

1. Create a RunPod Volume (50GB recommended)
2. Upload your MJAnime clips and metadata:
   ```
   /app/data/MJAnime/
   ├── metadata_final_clean_shots.json
   ├── clip1.mp4
   ├── clip2.mp4
   └── ...
   ```

## Step 5: Deploy Serverless Endpoint

1. Go to RunPod Dashboard → Serverless → Endpoints
2. Click "New Endpoint"
3. Select your template
4. Configure settings:
   - **GPU**: RTX 4090 (recommended for cost/performance)
   - **Active Workers**: 1-3 (based on expected load)
   - **Max Workers**: 10 (for auto-scaling)
   - **Request Timeout**: 300 seconds
   - **Container Timeout**: 600 seconds

## Step 6: Test Deployment

### Health Check
```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "input": {
      "endpoint": "health"
    }
  }'
```

### Generate Video
```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "input": {
      "endpoint": "generate",
      "script_text": "Your script content here",
      "script_name": "test_script",
      "variation_number": 1,
      "caption_style": "tiktok",
      "music_sync": true,
      "min_clip_duration": 2.5
    }
  }'
```

## API Endpoints

### Generate Video
- **Endpoint**: `generate`
- **Method**: POST
- **Input**:
  ```json
  {
    "endpoint": "generate",
    "script_text": "Your script content",
    "script_name": "script_name",
    "variation_number": 1,
    "caption_style": "tiktok",
    "music_sync": true,
    "min_clip_duration": 2.5
  }
  ```

### Health Check
- **Endpoint**: `health`
- **Method**: POST
- **Input**:
  ```json
  {
    "endpoint": "health"
  }
  ```

## Cost Optimization

1. **Use Spot Instances**: 60-91% cheaper than on-demand
2. **Auto-scaling**: Set max workers to handle traffic spikes
3. **Efficient Caching**: Pre-load models and data
4. **Batch Processing**: Process multiple requests together

## Monitoring

1. **RunPod Dashboard**: Monitor usage and costs
2. **Logs**: Check container logs for errors
3. **Metrics**: Track processing times and success rates

## Troubleshooting

### Common Issues

1. **Container won't start**: Check Docker logs
2. **GPU not available**: Verify CUDA installation
3. **File not found**: Check volume mounts and paths
4. **Memory issues**: Increase container disk size
5. **Timeout errors**: Increase request/container timeout

### Debug Commands

```bash
# Check GPU availability
nvidia-smi

# Check Python environment
python3 -c "import torch; print(torch.cuda.is_available())"

# Test video generation locally
python3 main.py test
```

## Scaling

- **Horizontal**: Increase max workers
- **Vertical**: Use more powerful GPU instances
- **Geographic**: Deploy in multiple regions

## Security

- Store API keys securely
- Use environment variables for secrets
- Implement request authentication
- Monitor for abuse patterns

## Next Steps

1. Set up monitoring and alerts
2. Implement request queuing
3. Add webhook notifications
4. Create client SDKs
5. Set up CI/CD pipeline