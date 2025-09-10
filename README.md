# NLBS-CVAE: Conditional Variational Autoencoder for Mammography Generation

A sophisticated deep learning project for generating high-quality Full-Field Digital Mammograms (FFDM) based on clinical conditions using Conditional Variational Autoencoders.

## Quick Start

### 🌟 Google Colab (Recommended for beginners)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FructueuxCODJIA/nlbs-cvae/blob/main/colab/notebooks/NLBS_CVAE_Training.ipynb)

**Perfect for experimentation and learning!**
- No local setup required
- Free GPU access
- Automatic environment setup
- Interactive training and generation

👉 **[See Colab Guide](colab/README.md)** for detailed instructions.

### 🖥️ Local/HPC Installation

### 1. Installation
```bash
# Clone the repository (if not already done)
cd /home/deadmaster/Documents/MammoGen/nlbs-cvae

# Install dependencies (already done)
pip install torch torchvision pandas pyyaml tqdm albumentations opencv-python scipy scikit-image pydicom tensorboard matplotlib seaborn
```

### 2. Test the Installation
```bash
# Test project structure
python test_structure.py

# Test model functionality
python test_model.py

# Test training pipeline
python test_training.py

# Run demo
python demo.py
```

### 3. Prepare Your Data
Update the data paths in `configs/training_config.yaml`:
```yaml
data:
  csv_path: "/path/to/your/metadata.csv"
  image_dir: "/path/to/your/images"
```

### 4. Start Training
```bash
python training/train_cvae.py
```

### 5. Monitor Training
```bash
tensorboard --logdir results/logs
```

## 📋 Project Structure

```
nlbs-cvae/
├── models/                 # Model architectures
│   ├── __init__.py
│   ├── cvae.py            # Main ConditionalVAE model
│   ├── layers.py          # Building blocks (ConvBlock, FiLM, etc.)
│   └── losses.py          # VAE loss functions
├── data/                  # Data processing
│   ├── __init__.py
│   ├── dataset.py         # MammographyDataset
│   ├── transforms.py      # Data augmentation
│   └── utils.py           # Patch extraction utilities
├── training/              # Training scripts
│   └── train_cvae.py      # Main training script
├── utils/                 # Utilities
│   ├── training_utils.py  # Training helpers
│   └── generate_samples.py # Sample generation
├── assessment/            # Evaluation
│   └── evaluate_model.py  # Model evaluation metrics
├── configs/               # Configuration files
│   └── training_config.yaml # Main configuration
├── test_*.py             # Test scripts
├── demo.py               # Demo script
└── README.md             # This file
```

## 🏗️ Architecture

### Model Components
- **Encoder**: 4 convolutional blocks (64→128→256→512 channels)
- **Latent Space**: 256-dimensional Gaussian distribution
- **Decoder**: 4 deconvolutional blocks with FiLM conditioning
- **Conditioning**: Multi-modal clinical attributes

### Conditioning System
The model generates images based on 5 clinical conditions:
- **View**: CC (craniocaudal) or MLO (mediolateral oblique)
- **Laterality**: Left or Right breast
- **Age**: Binned into 4 age groups
- **Cancer**: Presence/absence of cancer
- **False Positive**: Whether it's a false positive screening result

### Loss Function
```
L = L_rec + β * KL + λ_edge * L_edge
```
- **L1 reconstruction loss** for image fidelity
- **KL divergence** with annealing (β: 0→1 over 20 epochs)
- **Edge preservation loss** for micro-calcifications (λ_edge = 0.1)

## 🎯 Key Features

### Technical Innovations
- **FiLM Conditioning**: Feature-wise Linear Modulation for precise condition control
- **Edge-Aware Loss**: Preserves micro-calcifications crucial for mammography
- **Patch-Based Training**: Handles high-resolution images efficiently
- **Skip Connections**: Preserves fine details during reconstruction

### Training Features
- **KL Annealing**: Gradual introduction of KL loss for stable training
- **Comprehensive Logging**: TensorBoard integration with image galleries
- **Flexible Configuration**: YAML-based configuration system
- **Checkpoint Support**: Resume training from saved checkpoints

## 📊 Usage Examples

### Basic Model Usage
```python
from models import ConditionalVAE

# Create model
model = ConditionalVAE(
    in_channels=1,
    image_size=256,
    latent_dim=256,
    condition_embed_dim=128,
    encoder_channels=[64, 128, 256, 512],
    decoder_channels=[512, 256, 128, 64, 1]
)

# Generate samples
conditions = {
    'view': torch.tensor([0]),  # CC
    'laterality': torch.tensor([0]),  # Left
    'age_bin': torch.tensor([1]),  # Young-middle
    'cancer': torch.tensor([0]),  # No cancer
    'false_positive': torch.tensor([0])  # Not false positive
}

samples = model.sample(conditions, batch_size=1)
```

### Training Configuration
```yaml
# configs/training_config.yaml
model:
  latent_dim: 256
  condition_embed_dim: 128
  encoder:
    channels: [64, 128, 256, 512]
  decoder:
    channels: [512, 256, 128, 64, 1]

training:
  learning_rate: 1e-4
  num_epochs: 100
  loss:
    reconstruction_weight: 1.0
    kl_weight: 1.0
    edge_weight: 0.1
    kl_anneal_epochs: 20
```

## 🔧 Configuration

The project uses YAML configuration files. Key sections:

- **data**: Dataset paths and preprocessing parameters
- **model**: Architecture settings and dimensions
- **training**: Optimization and loss configuration
- **evaluation**: Validation and metrics settings
- **logging**: TensorBoard and checkpoint settings

## 📈 Evaluation

The project includes comprehensive evaluation metrics:
- **Image Quality**: FID, SSIM, PSNR, LPIPS
- **Medical Validity**: Clinical downstream task performance
- **Diversity**: Latent space traversals and condition control

## 🧪 Testing

The project includes several test scripts:

1. **test_structure.py**: Validates project structure and dependencies
2. **test_model.py**: Tests model architecture and forward pass
3. **test_training.py**: Tests training pipeline with dummy data
4. **demo.py**: Comprehensive demonstration of all features

## 📝 Data Format

Expected CSV format:
```csv
out_path,view,laterality,age_bin,cancer,false_positive,bbox,windowing
/path/to/image1.dcm,CC,L,1,0,0,"[x1,y1,x2,y2]","[center,width]"
/path/to/image2.dcm,MLO,R,2,1,0,"[x1,y1,x2,y2]","[center,width]"
```

## 🚀 Performance

- **Model Size**: ~104M parameters (configurable)
- **Training**: Supports both CPU and GPU training
- **Memory**: Optimized for patch-based processing
- **Speed**: Efficient inference with batch processing

## 🔬 Research Applications

This project is designed for:
- **Data Augmentation**: Generate synthetic mammograms for training
- **Conditional Generation**: Create images with specific clinical characteristics
- **Latent Space Analysis**: Study mammography feature representations
- **Medical AI Research**: Benchmark and evaluation datasets

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA/PyTorch Issues**: The project works with CPU-only PyTorch
2. **Memory Issues**: Reduce batch size or image resolution in config
3. **Data Loading**: Ensure CSV paths and image directories are correct
4. **Dependencies**: Run test scripts to verify installation

### Getting Help

1. Run `python test_structure.py` to check basic setup
2. Run `python demo.py` to see working examples
3. Check configuration in `configs/training_config.yaml`
4. Review logs in `results/logs/` directory

## 📄 License

This project is part of the MammoGen research initiative for advancing mammography AI.

## 🎉 Status

✅ **FULLY FUNCTIONAL** - The project is ready for use!

- All models implemented and tested
- Training pipeline working
- Evaluation metrics included
- Comprehensive documentation
- Demo scripts available

**Next Steps:**
1. Prepare your mammography dataset
2. Update configuration paths
3. Start training: `python training/train_cvae.py`
4. Monitor with TensorBoard: `tensorboard --logdir results/logs`

