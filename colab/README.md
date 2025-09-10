# NLBS-CVAE Google Colab Version

This directory contains the Google Colab adaptation of the NLBS-CVAE project for mammography generation.

## 🚀 Quick Start

### Option 1: Direct Colab Links (Recommended)
Click these links to open the notebooks directly in Google Colab:

- **Training**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FructueuxCODJIA/nlbs-cvae/blob/main/colab/notebooks/NLBS_CVAE_Training.ipynb)
- **Inference**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FructueuxCODJIA/nlbs-cvae/blob/main/colab/notebooks/NLBS_CVAE_Inference.ipynb)

### Option 2: Manual Setup
1. Open [Google Colab](https://colab.research.google.com/)
2. Upload the notebook files from `notebooks/` folder
3. Run all cells to start training/inference

## 📁 Directory Structure

```
colab/
├── notebooks/
│   ├── NLBS_CVAE_Training.ipynb    # Main training notebook
│   └── NLBS_CVAE_Inference.ipynb   # Inference and generation
├── configs/
│   └── colab_training_config.yaml  # Colab-optimized configuration
├── utils/
│   └── colab_helpers.py            # Colab-specific utilities
├── requirements_colab.txt          # Colab requirements
└── README.md                       # This file
```

## 🎯 Features

### Training Notebook (`NLBS_CVAE_Training.ipynb`)
- **🚀 Optimized for Colab**: GPU/TPU support, memory management
- **💾 Google Drive Integration**: Automatic data and checkpoint backup
- **📊 Real-time Monitoring**: TensorBoard integration
- **🔄 Session Management**: Handles Colab's runtime limitations
- **📈 Memory Optimization**: Automatic cache clearing and garbage collection

### Inference Notebook (`NLBS_CVAE_Inference.ipynb`)
- **🎯 Interactive Generation**: Widget-based parameter control
- **🖼️ Condition Control**: Generate mammograms with specific clinical conditions
- **🔍 Latent Space Exploration**: Interactive latent dimension visualization
- **📊 Batch Generation**: Generate multiple samples with different conditions

## 🔧 Configuration

The Colab version uses optimized settings for Google Colab's environment:

### Key Differences from HPC Version:
- **Reduced Model Size**: Smaller latent dimensions and channel counts
- **Lower Resolution**: 128x128 instead of 256x256 for faster training
- **Smaller Batch Size**: 8 instead of 16 to fit in Colab's memory
- **Shorter Training**: 50 epochs instead of 100
- **Frequent Checkpointing**: Every 2 epochs for session management
- **Memory Management**: Automatic cache clearing every 5 epochs

### Customization:
Edit `configs/colab_training_config.yaml` to modify:
- Model architecture parameters
- Training hyperparameters
- Data processing settings
- Hardware utilization

## 📊 Data Options

### 1. Demo Data (Default)
- Synthetic mammograms generated automatically
- Perfect for testing and learning
- No external data required

### 2. Upload Your Data
- Use Colab's file upload widget
- Support for ZIP files containing DICOM images
- Automatic extraction and organization

### 3. Google Drive Integration
- Store large datasets in Google Drive
- Automatic mounting and linking
- Persistent storage across sessions

## 🎮 Usage Examples

### Basic Training
```python
# The training notebook handles everything automatically:
# 1. Environment setup
# 2. Data preparation (demo or custom)
# 3. Model training with monitoring
# 4. Results backup to Google Drive
```

### Interactive Generation
```python
# Use the inference notebook for:
# - Interactive parameter control
# - Real-time generation
# - Latent space exploration
# - Batch generation with different conditions
```

### Custom Conditions
```python
conditions = {
    'view': 'CC',           # CC or MLO
    'laterality': 'Left',   # Left or Right
    'age_group': 'Middle',  # Young, Middle, Older, Senior
    'cancer': False,        # True/False
    'false_positive': False # True/False
}
```

## 📈 Performance Expectations

### Colab Free Tier:
- **GPU**: Tesla T4 (16GB VRAM)
- **RAM**: ~12GB
- **Training Time**: ~2-3 hours for 50 epochs
- **Session Limit**: 12 hours maximum

### Colab Pro:
- **GPU**: Tesla V100 or A100
- **RAM**: ~25GB
- **Training Time**: ~1-2 hours for 50 epochs
- **Session Limit**: 24 hours maximum

## 🔍 Monitoring and Debugging

### TensorBoard Integration
```python
%load_ext tensorboard
%tensorboard --logdir /content/results/logs
```

### Memory Monitoring
```python
from colab.utils.colab_helpers import monitor_resources
monitor_resources()  # Check GPU and RAM usage
```

### Automatic Backup
All results are automatically backed up to Google Drive:
- Model checkpoints
- Generated images
- Training logs
- Configuration files

## 🛠️ Troubleshooting

### Common Issues:

1. **Out of Memory**:
   - Reduce batch size in config
   - Lower image resolution
   - Enable gradient accumulation

2. **Session Timeout**:
   - Use frequent checkpointing
   - Enable automatic backup
   - Consider Colab Pro for longer sessions

3. **Slow Training**:
   - Ensure GPU is enabled
   - Use mixed precision training
   - Reduce model complexity

4. **Data Loading Issues**:
   - Check file paths in config
   - Verify Google Drive mounting
   - Use demo data for testing

### Getting Help:
1. Check the notebook outputs for error messages
2. Monitor resource usage with built-in tools
3. Refer to the main repository for detailed documentation
4. Use demo data to isolate issues

## 🔄 Migration from HPC

If you have an existing HPC setup, you can migrate to Colab:

1. **Export your data** to Google Drive
2. **Adjust the configuration** for Colab constraints
3. **Use the training notebook** with your data
4. **Scale up gradually** as needed

## 📚 Additional Resources

- [Google Colab Documentation](https://colab.research.google.com/notebooks/intro.ipynb)
- [PyTorch on Colab](https://pytorch.org/tutorials/beginner/colab.html)
- [Main NLBS-CVAE Repository](https://github.com/FructueuxCODJIA/nlbs-cvae)
- [TensorBoard Guide](https://www.tensorflow.org/tensorboard/get_started)

## 🤝 Contributing

To contribute to the Colab version:
1. Test your changes on Colab Free tier
2. Ensure compatibility with both GPU and CPU
3. Update documentation and examples
4. Submit pull requests to the main repository

## 📄 License

This Colab adaptation follows the same license as the main NLBS-CVAE project.

---

**Happy Training! 🎉**

For questions or issues, please refer to the main repository or create an issue on GitHub.