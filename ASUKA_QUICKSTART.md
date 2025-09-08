# 🚀 NLBS-CVAE Quick Start Guide for Asuka HPC

## Step 1: Transfer Project to Asuka

### On your local machine:
```bash
# Navigate to project directory
cd /home/deadmaster/Documents/MammoGen

# Create archive (manually, since tar command has issues)
# You can use your file manager to create a tar.gz of the nlbs-cvae folder
# Or run this in a separate terminal:
tar -czf nlbs-cvae-hpc.tar.gz nlbs-cvae/ --exclude='nlbs-cvae/.venv' --exclude='nlbs-cvae/__pycache__' --exclude='nlbs-cvae/**/__pycache__' --exclude='nlbs-cvae/results'

# Transfer to Asuka
scp nlbs-cvae-hpc.tar.gz user156@asuka.imsp-uac.org:/home/user156/
```

## Step 2: Setup on Asuka

### SSH to Asuka:
```bash
ssh -X user156@asuka.imsp-uac.org
```

### Extract and setup:
```bash
cd /home/user156
tar -xzf nlbs-cvae-hpc.tar.gz
cd nlbs-cvae
chmod +x setup_asuka.sh
./setup_asuka.sh
```

## Step 3: Prepare Your Data

### Copy your mammography data:
```bash
# Copy your CSV metadata file
cp /path/to/your/NLBSP-metadata.csv /home/user156/nlbs-cvae-data/

# Copy your image directory
cp -r /path/to/your/images /home/user156/nlbs-cvae-data/
```

### Update configuration:
```bash
# Edit the config file to match your data
nano configs/asuka_config.yaml

# Update these paths:
# csv_path: "/home/user156/nlbs-cvae-data/NLBSP-metadata.csv"
# image_dir: "/home/user156/nlbs-cvae-data/images"
```

## Step 4: Submit Training Job

### Check SLURM configuration:
```bash
# Review and edit if needed
nano train_asuka.slurm
```

### Submit job:
```bash
sbatch train_asuka.slurm
```

### Monitor job:
```bash
# Check job status
squeue -u user156

# Watch job output
tail -f slurm-JOBID.out

# Monitor training progress
tail -f /home/user156/nlbs-cvae-results/logs/training_*.log
```

## Step 5: Monitor Training

### Check GPU usage:
```bash
nvidia-smi  # If available
```

### Check results:
```bash
# List checkpoints
ls -la /home/user156/nlbs-cvae-results/checkpoints/

# Check TensorBoard logs
ls -la /home/user156/nlbs-cvae-results/logs/
```

## Step 6: Download Results (On Local Machine)

```bash
# Download results
scp -r user156@asuka.imsp-uac.org:/home/user156/nlbs-cvae-results ./asuka-results

# Download logs
scp user156@asuka.imsp-uac.org:/home/user156/nlbs-cvae/slurm-*.out ./asuka-results/

# View TensorBoard locally
cd asuka-results
tensorboard --logdir logs/
```

## 🔧 Troubleshooting

### If setup fails:
```bash
# Check available modules
module avail

# Check Python
python3 --version
which python3

# Check CUDA
nvidia-smi
```

### If training fails:
```bash
# Check data paths
ls -la /home/user156/nlbs-cvae-data/

# Check config
cat configs/asuka_config.yaml

# Check logs
tail -50 slurm-JOBID.err
```

### Common fixes:
```bash
# If PyTorch CUDA fails, install CPU version:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# If memory issues, reduce batch size in config:
# batch_size: 16  # or even 8
```

## 📊 Expected Training Time

- **Small dataset (1K images)**: 2-4 hours
- **Medium dataset (10K images)**: 8-12 hours  
- **Large dataset (100K images)**: 24-48 hours

## 🎯 Success Indicators

✅ Job starts without errors
✅ GPU utilization > 80% (if GPU available)
✅ Loss decreases over epochs
✅ Checkpoints are saved regularly
✅ Sample images look reasonable

## 📞 Need Help?

1. Check the SLURM output: `slurm-JOBID.out`
2. Check the error log: `slurm-JOBID.err`
3. Check training logs: `/home/user156/nlbs-cvae-results/logs/`
4. Test model locally: `python -c "from models import ConditionalVAE; print('OK')"`