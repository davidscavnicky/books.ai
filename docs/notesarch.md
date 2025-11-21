# AWS Architecture Notes

## Component Flow

### 1. User/Client
- Makes HTTP requests from browser/mobile app

### 2. ALB (Application Load Balancer)
AWS managed load balancer that:
- Distributes traffic across multiple API containers
- Handles SSL/TLS termination
- Provides health checks
- Auto-scales based on traffic

### 3. ECS Fargate (API Container)
Serverless container running your Flask API:
- No server management needed
- Auto-scales based on CPU/memory usage
- Contains two services:
  - **Flask API** (`scripts/api.py`) - handles HTTP endpoints
  - **Model Service** - loads TF-IDF and CF models from S3

## Data & Storage

### 4. S3 Bucket
Object storage for:
- Raw CSV data (`Books.csv`, `Ratings.csv`)
- Trained model artifacts (`.pkl` files)
- API reads models from here on startup

### 5. RDS (PostgreSQL)
Relational database for:
- User metadata
- Book metadata
- Rating history
- API logs

### 6. ElastiCache (Redis)
In-memory cache for:
- Popular recommendations (cached for fast access)
- Frequently requested book data
- Reduces database load

## Training Pipeline

### 7. Training (SageMaker/AWS Batch)
Scheduled job that:
- Runs `train_recommender.py` periodically (e.g., nightly)
- Pulls fresh data from S3/RDS
- Trains new models
- Uploads new `.pkl` files back to S3
- API can reload models without downtime

## Operations

### 8. CloudWatch
Monitoring service that:
- Collects metrics (API latency, error rates, traffic)
- Stores logs from all services
- Triggers alerts on errors

### 9. CI/CD (GitHub Actions)
Automated deployment:
- Runs tests on code push
- Builds Docker image
- Pushes to ECR (Elastic Container Registry)
- Updates ECS service with new image
- Zero-downtime rolling deployments

## Why This Architecture?

- **Scalable**: ECS auto-scales with traffic
- **Cost-effective**: Pay only for what you use (Fargate, S3)
- **Resilient**: ALB handles failover, multiple availability zones
- **Fast**: Redis caching reduces latency
- **Automated**: CI/CD + scheduled retraining
