# NLBSP Dataset Setup Guide for Google Colab

This guide will help you set up your NLBSP mammography dataset for training the NLBS-CVAE model in Google Colab.

## 🚀 Quick Start

### Option 1: Upload Data to Colab (Recommended for smaller datasets)

1. **Open the Training Notebook**:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FructueuxCODJIA/nlbs-cvae/blob/main/colab/notebooks/NLBS_CVAE_Training.ipynb)

2. **Run the setup cells** (1-4) to install requirements and mount Google Drive

3. **Navigate to "Option 4: NLBSP Dataset"** and uncomment the code

4. **Set data source to "upload"**:
   ```python
   data_source = "upload"  # This will prompt you to upload files
   ```

5. **Prepare your data** for upload:
   - Create a ZIP file containing:
     - `NLBSP-metadata.csv` (your metadata file)
     - `images/` folder with all your DICOM files organized as in your CSV

6. **Run the NLBSP setup cell** - it will:
   - Prompt you to upload your ZIP file
   - Extract and organize the data
   - Analyze your dataset
   - Create train/validation/test splits
   - Show sample images
   - Configure the model for real data

### Option 2: Use Google Drive (Recommended for larger datasets)

1. **Upload your data to Google Drive**:
   ```
   /content/drive/MyDrive/NLBS_Data/
   ├── NLBSP-metadata.csv
   └── images/
       └── abnormal/
           ├── 109430185503/
           │   ├── left/
           │   │   ├── CC/
           │   │   │   └── IM-0019-0002-0001.dcm
           │   │   └── MLO/
           │   │       └── IM-0021-0004-0001.dcm
           │   └── right/
           │       ├── CC/
           │       │   └── IM-0018-0001-0001.dcm
           │       └── MLO/
           │           └── IM-0020-0003-0001.dcm
           └── ... (other patients)
   ```

2. **In the notebook, set data source to "drive"**:
   ```python
   data_source = "drive"  # This will use data from Google Drive
   ```

3. **Run the NLBSP setup cell** - it will automatically find and process your data

## 📊 What the Setup Does

The NLBSP setup process will:

1. **📤 Data Loading**: Upload or link your data from Google Drive
2. **🔍 Analysis**: Analyze your dataset and show statistics:
   - Total images and unique patients
   - Distribution of views (CC/MLO) and laterality (L/R)
   - Cancer vs normal cases
   - Age distribution
   - File existence check

3. **🔄 Preprocessing**: 
   - Convert Windows paths to Unix format
   - Create age bins for training
   - Add numeric encodings for categorical variables
   - Extract patient IDs

4. **📊 Data Splits**: Create train/val/test splits ensuring:
   - Patient-level separation (no patient appears in multiple splits)
   - Balanced distribution across splits

5. **🖼️ Visualization**: Show sample images from your dataset

6. **⚙️ Configuration**: Set up the model configuration optimized for real mammography data:
   - Higher resolution (256x256)
   - Larger latent dimensions
   - Appropriate loss weights
   - Memory optimization for Colab

## 🔧 Configuration Details

When using real NLBSP data, the system automatically switches to `colab_real_data_config.yaml` which includes:

### Model Architecture:
- **Latent Dimension**: 256 (vs 128 for demo)
- **Image Resolution**: 256x256 (vs 128x128 for demo)
- **Encoder/Decoder Channels**: [32, 64, 128, 256, 512]

### Training Settings:
- **Batch Size**: 4 (optimized for Colab memory)
- **Epochs**: 100 (more training for real data)
- **Learning Rate**: 0.0002
- **Mixed Precision**: Enabled for memory efficiency

### Data Processing:
- **Age Binning**: Automatic based on your data range
- **Augmentation**: Conservative (medical image appropriate)
- **Normalization**: Proper DICOM preprocessing

## 📈 Expected Performance

### Training Time (Colab Free Tier):
- **Setup**: ~5-10 minutes
- **Training**: ~4-6 hours for 100 epochs
- **Total Session**: Can complete in one 12-hour session

### Memory Usage:
- **GPU Memory**: ~10-12GB (fits in T4)
- **RAM**: ~8-10GB
- **Storage**: ~2-5GB depending on dataset size

## 🛠️ Troubleshooting

### Common Issues:

1. **"Out of Memory" Error**:
   ```python
   # Reduce batch size in config
   config['data']['batch_size'] = 2  # Instead of 4
   ```

2. **"File Not Found" Error**:
   - Check that your DICOM files match the paths in CSV
   - Ensure Windows paths are converted (handled automatically)

3. **"Session Timeout"**:
   - The notebook saves checkpoints every 5 epochs
   - You can resume training from the last checkpoint

4. **"Upload Failed"**:
   - Try smaller ZIP files (<2GB)
   - Use Google Drive for larger datasets

### Getting Help:

1. **Check the notebook outputs** for detailed error messages
2. **Monitor resource usage** with the built-in tools
3. **Use demo data first** to test the setup
4. **Refer to the main README** for additional troubleshooting

## 🎯 Next Steps

After successful setup:

1. **Monitor Training**: Use TensorBoard to track progress
2. **Generate Samples**: Use the inference notebook for generation
3. **Backup Results**: Automatic backup to Google Drive
4. **Experiment**: Try different configurations and hyperparameters

## 📚 Additional Resources

- [Main Project README](../README.md)
- [Colab General Guide](README.md)
- [Inference Notebook](notebooks/NLBS_CVAE_Inference.ipynb)
- [Configuration Reference](configs/colab_real_data_config.yaml)

---

**Happy Training! 🎉**

Your NLBSP mammography dataset is now ready for state-of-the-art conditional VAE training in Google Colab!