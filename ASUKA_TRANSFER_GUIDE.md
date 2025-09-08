# 🚀 Transfer Guide for Asuka HPC

## Step 1: Create Project Archive (Local Machine)

Since the automated tar command isn't working, create the archive manually:

### Option A: Using File Manager
1. Open file manager and navigate to `/home/deadmaster/Documents/MammoGen/`
2. Right-click on `nlbs-cvae` folder
3. Select "Compress" or "Create Archive"
4. Name it `nlbs-cvae-hpc.tar.gz`
5. Exclude these folders/files:
   - `.venv/`
   - `__pycache__/`
   - `results/`

### Option B: Using Terminal (in a new terminal without venv)
```bash
cd /home/deadmaster/Documents/MammoGen
tar -czf nlbs-cvae-hpc.tar.gz nlbs-cvae/ \
    --exclude='nlbs-cvae/.venv' \
    --exclude='nlbs-cvae/__pycache__' \
    --exclude='nlbs-cvae/**/__pycache__' \
    --exclude='nlbs-cvae/results'
```

## Step 2: Transfer to Asuka

```bash
# Transfer the archive
scp nlbs-cvae-hpc.tar.gz user156@asuka.imsp-uac.org:/home/user156/

# Also transfer your data (if not already on HPC)
scp "/media/deadmaster/New Volume/NLBS Data/NLBSP-metadata.csv" user156@asuka.imsp-uac.org:/home/user156/
scp -r "/media/deadmaster/New Volume/NLBS Data/images" user156@asuka.imsp-uac.org:/home/user156/nlbs-data/
```

## Step 3: Setup on Asuka

You're already connected to Asuka, so run these commands:

```bash
# Extract the project
cd /home/user156
tar -xzf nlbs-cvae-hpc.tar.gz
cd nlbs-cvae

# Copy the custom setup script (from the files I created)
# You'll need to create this file on Asuka with the content I provided

# Make it executable and run
chmod +x setup_asuka_custom.sh
./setup_asuka_custom.sh
```

## Step 4: Prepare Data

```bash
# Create data directory
mkdir -p /home/user156/nlbs-cvae-data

# Copy your data files
cp /home/user156/NLBSP-metadata.csv /home/user156/nlbs-cvae-data/
cp -r /home/user156/nlbs-data/images /home/user156/nlbs-cvae-data/

# Verify data
ls -la /home/user156/nlbs-cvae-data/
head -3 /home/user156/nlbs-cvae-data/NLBSP-metadata.csv
```

## Step 5: Submit Job

```bash
# Check the job script
cat train_asuka.slurm

# Submit the job
sbatch train_asuka.slurm

# Check job status
squeue -u user156

# Monitor output
tail -f slurm-JOBID.out  # Replace JOBID with actual job ID
```

## Available Modules on Asuka

Based on your `module avail` output:
- ✅ `gcc/gcc-12.0` - C++ compiler
- ✅ `Python/python-3.11.6` - Python 3.11
- ✅ `Python/python-3.8.18` - Python 3.8 (alternative)
- ❌ No CUDA modules visible (might be CPU-only cluster)

## Expected Setup Process

1. **Module Loading**: Uses gcc-12.0 and Python-3.11.6
2. **Virtual Environment**: Creates isolated Python environment
3. **PyTorch Installation**: Will install CPU version if no GPU detected
4. **Dependencies**: Installs all required packages
5. **Configuration**: Creates Asuka-specific config files
6. **Testing**: Verifies model can be created and run

## Troubleshooting

### If setup fails:
```bash
# Check Python
python3 --version
which python3

# Check available space
df -h /home/user156

# Check modules
module list
```

### If training fails:
```bash
# Check data
ls -la /home/user156/nlbs-cvae-data/
wc -l /home/user156/nlbs-cvae-data/NLBSP-metadata.csv

# Check logs
tail -50 slurm-JOBID.err
```

## Next Steps After This Guide

1. Create the archive on your local machine
2. Transfer to Asuka
3. Extract and run setup
4. Copy your data
5. Submit the training job
6. Monitor progress

Let me know when you've completed the transfer and I can help with the next steps!