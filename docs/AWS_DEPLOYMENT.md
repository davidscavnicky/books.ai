# AWS Deployment Architecture

## Overview
This document describes the production deployment architecture for the BooksAI recommendation system on AWS.

## Architecture Diagrams

### 1. AWS Deployment Architecture
Shows the complete production infrastructure with:
- **ALB (Application Load Balancer)**: Entry point for all client requests
- **ECS Fargate**: Containerized API service (scripts/api.py)
- **S3**: Storage for raw data and trained model artifacts
- **SageMaker/Batch**: Scheduled model training (scripts/train_recommender.py)
- **RDS**: PostgreSQL for metadata and user interaction logs
- **ElastiCache (Redis)**: Caching layer for popular recommendations
- **CloudWatch**: Monitoring and logging
- **GitHub Actions**: CI/CD pipeline

### 2. Data Pipeline Flow
Shows the end-to-end ML pipeline:
1. **Data Loading**: Kaggle dataset (Books.csv, Ratings.csv) via train_recommender.py
2. **Data Transformation**: Clean text, filter ratings, build sparse matrices
3. **Model Training**: Popularity baseline, TF-IDF content vectors, Item-Item CF
4. **API**: Flask REST endpoints serving predictions
5. **Front-End**: User interface (currently curl/Postman, future: React/Vue)

## Component Details

### API Service (ECS Fargate)
- **Container**: Python 3.12 with Flask + dependencies
- **Scale**: Auto-scaling 2-10 tasks based on CPU/memory
- **Health**: `/healthz` endpoint monitored by ALB
- **Environment**: Load models from S3 at startup

### Training Pipeline (SageMaker/Batch)
- **Schedule**: Daily/weekly via EventBridge
- **Input**: Raw CSV from S3
- **Output**: Model artifacts (tfidf_vectorizer.pkl, books_df.pkl) to S3
- **Notification**: SNS alert on success/failure

### Data Storage
- **S3 Buckets**:
  - `booksai-raw-data/`: Kaggle CSV files
  - `booksai-models/`: Trained model artifacts
- **RDS**: User ratings, book metadata, API logs

### Caching Strategy
- **ElastiCache (Redis)**:
  - Cache popular recommendations (TTL: 1 hour)
  - Cache book metadata lookups (TTL: 24 hours)
  - Reduce load on API during peak traffic

## Deployment Steps

### Prerequisites
```bash
# Install AWS CLI
pip install awscli
aws configure

# Install Docker
brew install docker
```

### 1. Build and Push Docker Image
```bash
# Build API container
docker build -t booksai-api -f Dockerfile .

# Tag and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker tag booksai-api:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/booksai-api:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/booksai-api:latest
```

### 2. Deploy Infrastructure (Terraform/CloudFormation)
```bash
# Example Terraform commands
cd terraform/
terraform init
terraform plan
terraform apply
```

### 3. Upload Data and Models to S3
```bash
# Upload Kaggle data
aws s3 cp data/Books.csv s3://booksai-raw-data/Books.csv
aws s3 cp data/Ratings.csv s3://booksai-raw-data/Ratings.csv

# Upload trained models
aws s3 cp models/tfidf_vectorizer.pkl s3://booksai-models/tfidf_vectorizer.pkl
aws s3 cp models/books_df.pkl s3://booksai-models/books_df.pkl
```

### 4. Deploy ECS Service
```bash
# Update ECS service with new task definition
aws ecs update-service --cluster booksai-cluster --service booksai-api --force-new-deployment
```

## CI/CD Pipeline (GitHub Actions)

```yaml
# .github/workflows/deploy.yml
name: Deploy to AWS

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      - name: Build and push Docker image
        run: |
          docker build -t booksai-api .
          docker tag booksai-api:latest $ECR_REGISTRY/booksai-api:latest
          docker push $ECR_REGISTRY/booksai-api:latest
      - name: Deploy to ECS
        run: |
          aws ecs update-service --cluster booksai-cluster --service booksai-api --force-new-deployment
```

## Monitoring and Alerts

### CloudWatch Metrics
- API request count and latency (p50, p95, p99)
- ECS task CPU/memory utilization
- ALB target health status
- Cache hit rate (Redis)

### Alarms
- High API error rate (> 5%)
- ECS task failures
- Training job failures
- S3 bucket access errors

## Cost Optimization
- **ECS Fargate Spot**: Use spot instances for non-critical tasks
- **S3 Intelligent Tiering**: Auto-move old data to cheaper storage
- **ElastiCache Reserved**: Reserve cache nodes for 1-year commitment
- **Auto-scaling**: Scale down during off-peak hours

## Security
- **IAM Roles**: Least-privilege access for ECS tasks
- **VPC**: Private subnets for RDS and ElastiCache
- **Secrets Manager**: Store API keys and DB credentials
- **WAF**: Protect ALB from common attacks

## Disaster Recovery
- **RDS Automated Backups**: Daily snapshots (7-day retention)
- **S3 Versioning**: Enable on model artifacts bucket
- **Multi-AZ**: Deploy RDS and ElastiCache across AZs
- **Cross-Region Replication**: Replicate critical S3 buckets

## Future Enhancements
1. **Real-time Retraining**: Implement streaming pipeline (Kinesis + Lambda)
2. **A/B Testing**: Deploy multiple model versions and compare performance
3. **GraphQL API**: Replace REST with GraphQL for flexible queries
4. **React Front-End**: Deploy static site on S3 + CloudFront
5. **Personalization**: Add user embeddings and deep learning models (SageMaker)
