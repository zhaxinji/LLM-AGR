# LLM-AGR: Large Language Model Augmented Graph Representation Learning for Recommendation


## Detailed Project

### Configure Model

Select or modify YAML configuration files in the `config/models_config/` directory. Configuration files include:
- Optimizer settings (optimizer)
- Training parameters (train)
- Testing parameters and evaluation metrics (test)
- Dataset settings (data)
- Model parameters (model)

### Prepare Dataset

Place datasets in the `data/` directory with the following requirements:
- User-item interaction data
- Items/Users content information (for enhanced representation)


### View Results

Training logs are saved in the `log/` directory, and model checkpoints are saved in the `checkpoint/` directory.

## Project Structure

- `config/`: Configuration files and configuration management
- `models/`: Model implementations
  - `general_cf/`: Collaborative filtering model implementations
- `trainer/`: Training and evaluation related code
- `load_data/`: Data loading and preprocessing
- `data/`: Dataset storage
- `log/`: Training logs
- `checkpoint/`: Model saving

## Supported Models

- Base models: LightGCN, SimGCL, SGL, BiGCF
- LLM-enhanced models: LightGCN-AGR, SimGCL-AGR, SGL-AGR, BiGCF-AGR

## Supported Datasets

- Amazon
- Yelp
- Movie


## Quick Start

```bash
# Clone repository
git clone https://github.com/zhaxinji/LLM-AGR.git
cd LLM-AGR

python main.py
```

