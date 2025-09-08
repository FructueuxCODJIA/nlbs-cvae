# HPC Setup Guide for NLBS-CVAE

## Step 1: Package Your Project

### 1.1 Create a clean project archive
```bash
# On your local machine
cd /home/deadmaster/Documents/MammoGen
tar -czf nlbs-cvae.tar.gz nlbs-cvae/ --exclude='nlbs-cvae/.venv' --exclude='nlbs-cvae/__pycache__' --exclude='nlbs-cvae/**/__pycache__' --exclude='nlbs-cvae/results' --exclude='nlbs-cvae/test_*.py'
```

### 1.2 Create requirements file
```bash
cd nlbs-cvae
pip freeze > requirements.txt
```

## Step 2: Transfer to HPC

### 2.1 Upload project
```bash
# Replace with your HPC details
scp nlbs-cvae.tar.gz username@hpc-cluster.edu:/home/username/
```

### 2.2 SSH to HPC and extract
```bash
ssh username@hpc-cluster.edu
cd /home/username
tar -xzf nlbs-cvae.tar.gz
cd nlbs-cvae
```

## Step 3: HPC Environment Setup

### 3.1 Load modules (adapt to your HPC)
```bash
# Common HPC modules - adjust for your system
module load python/3.9
module load cuda/11.8
module load gcc/9.3.0
```

### 3.2 Create virtual environment
```bash
python -m venv venv
source venv/bin/activate
```

### 3.3 Install dependencies
```bash
# Install PyTorch for your CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install pandas pyyaml tqdm albumentations opencv-python scipy scikit-image pydicom tensorboard matplotlib seaborn
```

## Step 4: Configure for HPC

### 4.1 Update data paths in config
Edit `configs/training_config.yaml`:
```yaml
data:
  csv_path: "/path/to/your/hpc/data/NLBSP-metadata.csv"
  image_dir: "/path/to/your/hpc/data/images"
  batch_size: 32  # Increase for HPC
  num_workers: 8  # Adjust for HPC cores

hardware:
  device: "cuda"  # Use GPU on HPC

output:
  results_dir: "/path/to/your/hpc/results"
```

### 4.2 Create HPC job script
See the SLURM script below.

## Step 5: Submit Job

### 5.1 Test installation
```bash
# Quick test
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python test_model.py
```

### 5.2 Submit training job
```bash
sbatch train_job.slurm
```

### 5.3 Monitor job
```bash
squeue -u $USER
tail -f slurm-JOBID.out
```

## Step 6: Monitor and Retrieve Results

### 6.1 Check progress
```bash
# View logs
tail -f results/logs/training.log

# Check TensorBoard logs
ls results/logs/events.out.tfevents.*
```

### 6.2 Download results
```bash
# On your local machine
scp -r username@hpc-cluster.edu:/home/username/nlbs-cvae/results ./
```