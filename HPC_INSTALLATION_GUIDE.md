# HPC Installation Guide for NLBS-CVAE

## Problem Diagnosis

The installation failures you encountered are due to:

1. **Network connectivity issues**: The HPC system cannot reach PyTorch download servers
2. **Module system differences**: The available modules don't match the script expectations
3. **Outdated pip version**: The system pip is too old to handle modern package installations

## Solution Options

### Option 1: Updated Setup Script (Recommended)

Use the updated `setup_hpc.sh` script which includes:

```bash
./setup_hpc.sh
```

**Key improvements:**
- Uses correct module names (`Python/python-3.11.6` or `Python/python-3.8.18`)
- Fallback installation methods (PyTorch index → PyPI → offline)
- Better error handling and network connectivity checks
- Automatic CPU/GPU detection and configuration

### Option 2: Offline Installation

If network issues persist, use the offline setup:

```bash
./setup_hpc_offline.sh
```

This creates a minimal environment that can be completed later when packages are available.

### Option 3: Manual Installation

1. **Check environment first:**
   ```bash
   ./check_hpc_environment.sh
   ```

2. **Load correct modules:**
   ```bash
   module load Python/python-3.11.6  # or Python/python-3.8.18
   module load gcc/gcc-12.0
   ```

3. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   ```

4. **Install PyTorch (try in order):**
   ```bash
   # Try PyTorch index
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   
   # If that fails, try PyPI
   pip install torch torchvision torchaudio
   
   # If that fails, install minimal requirements
   pip install numpy pyyaml tqdm
   ```

## Network Connectivity Solutions

### If PyTorch servers are unreachable:

1. **Download on local machine with internet:**
   ```bash
   # On your local machine:
   pip download torch torchvision torchaudio --dest pytorch_packages/
   pip download pandas pyyaml tqdm albumentations opencv-python scipy scikit-image pydicom tensorboard matplotlib seaborn --dest other_packages/
   
   # Transfer to HPC:
   scp -r pytorch_packages/ other_packages/ user156@asuka.imsp-uac.org:~/nlbs-cvae/
   
   # On HPC:
   pip install --find-links pytorch_packages/ torch torchvision torchaudio
   pip install --find-links other_packages/ pandas pyyaml tqdm albumentations opencv-python scipy scikit-image pydicom tensorboard matplotlib seaborn
   ```

2. **Use system packages if available:**
   ```bash
   # Check what's available system-wide
   python3 -c "import sys; print(sys.path)"
   
   # Try installing with --user flag
   pip install --user torch torchvision torchaudio
   ```

## Configuration for CPU-only Training

Since no CUDA modules were found, configure for CPU training:

1. **Use CPU configuration:**
   ```bash
   cp configs/training_config.yaml configs/cpu_training_config.yaml
   ```

2. **Edit the config to use CPU:**
   ```yaml
   # In configs/cpu_training_config.yaml
   device: "cpu"
   batch_size: 8  # Smaller batch size for CPU
   num_workers: 2  # Fewer workers for CPU
   ```

3. **Create CPU SLURM script:**
   ```bash
   cp train_job.slurm train_cpu.slurm
   ```
   
   Edit `train_cpu.slurm`:
   ```bash
   #SBATCH --partition=cpu  # Use CPU partition
   #SBATCH --cpus-per-task=8
   # Remove any GPU-related SBATCH directives
   ```

## Testing Your Installation

1. **Check environment:**
   ```bash
   ./check_hpc_environment.sh
   ```

2. **Test PyTorch:**
   ```bash
   source venv/bin/activate
   python -c "import torch; print('PyTorch version:', torch.__version__)"
   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   ```

3. **Test model creation:**
   ```bash
   python -c "
   from models import ConditionalVAE
   import torch
   model = ConditionalVAE(in_channels=1, image_size=256, latent_dim=128, condition_embed_dim=64, encoder_channels=[32, 64, 128, 256], decoder_channels=[256, 128, 64, 32, 1])
   print('Model created successfully!')
   "
   ```

## Troubleshooting

### Common Issues:

1. **"Module not found" errors:**
   - Check available modules: `module avail`
   - Use exact module names from the output

2. **"Connection failed" errors:**
   - Use offline installation method
   - Download packages separately and transfer

3. **"Permission denied" errors:**
   - Ensure you're in your home directory
   - Check file permissions: `ls -la`

4. **"Virtual environment creation failed":**
   - Try with different Python version
   - Use `--system-site-packages` flag

### Getting Help:

1. **Check system status:**
   ```bash
   ./check_hpc_environment.sh
   ```

2. **Contact HPC support with:**
   - Output of the check script
   - Specific error messages
   - Your username and project directory

## Next Steps After Successful Installation

1. **Copy your data:**
   ```bash
   # Copy to: /home/user156/NLBS Data/
   ```

2. **Submit training job:**
   ```bash
   # For CPU training:
   sbatch train_cpu.slurm
   
   # Monitor:
   squeue -u user156
   ```

3. **Check results:**
   ```bash
   # Results will be in: /home/user156/nlbs-cvae-results/
   ```

## Summary

The updated scripts should resolve the network and module issues. If problems persist, use the offline installation method and manually transfer packages from a machine with internet access.