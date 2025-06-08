# Azure ML Setup Guide

This guide shows how to set up Azure ML for cloud training when you need to scale beyond local resources.

## 🎯 When to Use Azure Training

Use Azure ML when:
- **Dataset size > 500K samples** (local training becomes slow)
- **Training time > 2 hours** (want to use cloud resources)
- **Need distributed training** (multi-GPU/multi-node)
- **Want cost optimization** (spot instances save 60-90%)

## 📋 Prerequisites

1. **Azure Subscription** with Azure ML workspace
2. **Azure CLI** installed and logged in
3. **Resource Group** and **ML Workspace** created

## 🚀 Quick Setup

### 1. Install Azure Dependencies
```bash
make install-azure
# Or manually: pip install azure-ai-ml azure-identity
```

### 2. Set Environment Variables
```bash
export AZURE_SUBSCRIPTION_ID="your-subscription-id"
export AZURE_RESOURCE_GROUP="your-resource-group"  
export AZURE_WORKSPACE_NAME="your-workspace-name"
```

### 3. Test Connection
```bash
make test-azure
```

### 4. Run Cloud Training
```bash
# Small dataset (100K samples)
make train-azure-small

# Large dataset (1M samples) 
make train-azure-large
```

## 💰 Cost Optimization

Our Azure integration includes automatic cost optimization:

- **Spot Instances**: 60-90% cheaper than regular instances
- **Auto-scaling**: Right-sized compute based on dataset
- **Time limits**: Prevents runaway costs
- **Cost estimates**: Shows expected costs before submission

### Instance Types and Costs

| Dataset Size | Instance | vCPUs | RAM | Cost/Hour | Spot Cost |
|-------------|----------|-------|-----|-----------|-----------|
| < 50K       | F2s_v2   | 2     | 4GB | $0.10     | $0.03     |
| < 500K      | F4s_v2   | 4     | 8GB | $0.19     | $0.06     |
| < 2M        | F8s_v2   | 8     | 16GB| $0.39     | $0.12     |
| > 2M        | D8s_v3   | 8     | 32GB| $0.44     | $0.13     |

## 🏗️ Azure ML Workspace Setup

If you don't have an Azure ML workspace:

### Option 1: Azure Portal (GUI)
1. Go to [Azure Portal](https://portal.azure.com)
2. Create Resource Group
3. Create Machine Learning workspace
4. Note subscription ID, resource group, and workspace name

### Option 2: Azure CLI (Command Line)
```bash
# Login
az login

# Create resource group
az group create --name "ml-pipeline-rg" --location "eastus"

# Create ML workspace  
az ml workspace create --name "ml-pipeline-ws" --resource-group "ml-pipeline-rg"

# Get workspace details
az ml workspace show --name "ml-pipeline-ws" --resource-group "ml-pipeline-rg"
```

## 📊 Monitoring Training Jobs

### View in Azure ML Studio
- Jobs automatically open Azure ML Studio URL
- Monitor progress, logs, and metrics in real-time
- Download trained models when complete

### Command Line Monitoring
```bash
# List recent jobs
az ml job list --max-results 5

# Get job status
az ml job show --name "your-job-name"

# Stream job logs
az ml job stream --name "your-job-name"
```

## 📥 Downloading Trained Models

Models are automatically available in Azure ML. To download:

```python
from training.azure_integration.submit_job import AzureTrainingSubmitter

submitter = AzureTrainingSubmitter(config)
submitter.download_model("job-name", "./models")
```

## 🔧 Advanced Configuration

### Custom Compute Clusters
Create dedicated compute for consistent performance:

```bash
# Create compute cluster
az ml compute create --name "gpu-cluster" \
  --type amlcompute \
  --min-instances 0 \
  --max-instances 4 \
  --size "Standard_NC6s_v3"
```

### Environment Variables for CI/CD
For automated deployments, set environment variables:

```bash
# In your CI/CD pipeline
export AZURE_SUBSCRIPTION_ID="${{ secrets.AZURE_SUBSCRIPTION_ID }}"
export AZURE_RESOURCE_GROUP="${{ secrets.AZURE_RESOURCE_GROUP }}"
export AZURE_WORKSPACE_NAME="${{ secrets.AZURE_WORKSPACE_NAME }}"
```

## 🚨 Troubleshooting

### Common Issues

**Authentication Error**
```bash
# Re-login to Azure
az login --tenant "your-tenant-id"
```

**Quota Exceeded**
- Request quota increase in Azure Portal
- Use smaller instances (F2s_v2 instead of F8s_v2)
- Use different regions (try "westus2", "eastus")

**Spot Instance Preemption**
- Jobs automatically retry with regular instances
- Use shorter durations (4h max recommended)
- Monitor job progress closely

## 💡 Best Practices

1. **Start Small**: Test with small datasets first
2. **Use Spot Instances**: 60-90% cost savings with minimal risk
3. **Monitor Costs**: Check Azure Cost Management regularly
4. **Set Spending Limits**: Configure spending alerts
5. **Clean Up**: Delete unused compute and storage

## 🔗 Useful Links

- [Azure ML Documentation](https://docs.microsoft.com/en-us/azure/machine-learning/)
- [Azure ML Python SDK](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/)
- [Azure Cost Calculator](https://azure.microsoft.com/en-us/pricing/calculator/)
- [Azure ML Pricing](https://azure.microsoft.com/en-us/pricing/details/machine-learning/)