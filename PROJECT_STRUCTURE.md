# NLBS-CVAE - Arborescence Complète du Projet

## Vue d'ensemble
Projet de génération d'images mammographiques utilisant un Conditional Variational Autoencoder (CVAE) optimisé pour l'entraînement sur HPC.

```
nlbs-cvae/
├── 📁 assessment/                    # Évaluation et métriques
│   └── evaluate_model.py            # Script d'évaluation des modèles
│
├── 📁 configs/                      # Configurations d'entraînement
│   ├── training_config.yaml         # Configuration locale standard
│   └── hpc_training_config.yaml     # Configuration optimisée HPC
│
├── 📁 data/                         # Gestion des données
│   ├── __init__.py                  # Module data
│   ├── dataset.py                   # Dataset NLBS mammographique
│   ├── transforms.py                # Transformations d'images
│   └── utils.py                     # Utilitaires de données
│
├── 📁 envs/                         # Environnements
│   └── requirements.yaml            # Dépendances Conda
│
├── 📁 models/                       # Architecture du modèle
│   ├── __init__.py                  # Module models
│   ├── cvae.py                      # Conditional VAE principal
│   ├── layers.py                    # Couches personnalisées
│   └── losses.py                    # Fonctions de perte
│
├── 📁 results/                      # Résultats d'entraînement
│   ├── galleries/                   # Galeries d'images générées
│   └── logs/                        # Logs d'entraînement
│
├── 📁 training/                     # Scripts d'entraînement
│   └── train_cvae.py                # Script principal d'entraînement
│
├── 📁 utils/                        # Utilitaires généraux
│   ├── __init__.py                  # Module utils
│   ├── generate_samples.py          # Génération d'échantillons
│   └── training_utils.py            # Utilitaires d'entraînement
│
├── 📄 DOCUMENTATION/                # Documentation complète
│   ├── README.md                    # Documentation principale
│   ├── NLBS_CVAE_Architecture_Guide.md  # Guide architecture
│   ├── HPC_INSTALLATION_GUIDE.md   # Guide installation HPC
│   ├── ASUKA_TRANSFER_GUIDE.md      # Guide transfert Asuka
│   ├── ASUKA_QUICKSTART.md          # Démarrage rapide Asuka
│   └── hpc_setup.md                 # Configuration HPC
│
├── 🔧 SCRIPTS HPC/                  # Scripts de déploiement HPC
│   ├── setup_hpc.sh                # Installation HPC principale
│   ├── setup_hpc_offline.sh        # Installation hors ligne
│   ├── setup_asuka.sh               # Installation Asuka
│   ├── setup_asuka_custom.sh        # Installation Asuka personnalisée
│   ├── check_hpc_environment.sh     # Vérification environnement
│   ├── check_hpc_readiness.sh       # Vérification préparation
│   ├── transfer_to_hpc.sh           # Transfert vers HPC
│   ├── transfer_to_asuka.sh         # Transfert vers Asuka
│   ├── download_results.sh          # Téléchargement résultats
│   └── monitor_hpc.sh               # Monitoring HPC
│
├── ⚙️ SLURM JOBS/                   # Scripts de soumission
│   ├── train_job.slurm              # Job SLURM standard
│   └── train_asuka.slurm            # Job SLURM Asuka
│
├── 🧪 TESTS/                        # Scripts de test
│   ├── demo.py                      # Démonstration complète
│   ├── quick_train.py               # Entraînement rapide
│   ├── test_model.py                # Test du modèle
│   ├── test_dataset.py              # Test du dataset
│   ├── test_structure.py            # Test de la structure
│   └── test_training.py             # Test d'entraînement
│
├── 📦 REQUIREMENTS/                 # Dépendances
│   ├── requirements.txt             # Dépendances standard
│   └── requirements_hpc.txt         # Dépendances HPC
│
└── 📁 .venv/                        # Environnement virtuel local
    └── [environnement Python local]
```

## Détails des Composants

### 🏗️ Architecture du Modèle (`models/`)

#### `cvae.py` - Conditional VAE Principal
- **ConditionalVAE**: Classe principale du modèle
- **Encoder**: Réseau encodeur avec convolutions
- **Decoder**: Réseau décodeur avec convolutions transposées
- **Conditioning**: Injection de conditions via FiLM
- **Reparameterization**: Trick de reparamétrisation VAE

#### `layers.py` - Couches Personnalisées
- **FiLMLayer**: Feature-wise Linear Modulation
- **ResidualBlock**: Blocs résiduels
- **AttentionBlock**: Mécanismes d'attention
- **GroupNorm**: Normalisation par groupes

#### `losses.py` - Fonctions de Perte
- **VAELoss**: Perte combinée reconstruction + KL
- **ReconstructionLoss**: Perte de reconstruction (MSE/L1)
- **KLDivergenceLoss**: Divergence KL avec annealing
- **EdgeLoss**: Perte de préservation des contours

### 📊 Gestion des Données (`data/`)

#### `dataset.py` - Dataset NLBS
- **NLBSDataset**: Dataset principal pour mammographies
- **Preprocessing**: Normalisation et redimensionnement
- **Filtering**: Filtrage par fraction de premier plan
- **Metadata**: Gestion des métadonnées cliniques

#### `transforms.py` - Transformations
- **Augmentations**: Rotation, flip, contraste, bruit
- **Normalization**: Normalisation des intensités
- **Resizing**: Redimensionnement adaptatif
- **Patching**: Extraction de patches

#### `utils.py` - Utilitaires de Données
- **Data loading**: Chargement efficace des données
- **Validation**: Validation des données
- **Statistics**: Calcul de statistiques

### 🎯 Entraînement (`training/`)

#### `train_cvae.py` - Script Principal
- **Training loop**: Boucle d'entraînement complète
- **Validation**: Évaluation périodique
- **Checkpointing**: Sauvegarde des modèles
- **Logging**: Logs détaillés avec TensorBoard
- **Mixed precision**: Support AMP pour efficacité

### 🔧 Configuration (`configs/`)

#### `training_config.yaml` - Configuration Locale
```yaml
# Configuration pour développement local
data:
  batch_size: 16
  resolution: 256
model:
  latent_dim: 128
training:
  num_epochs: 100
  learning_rate: 1e-4
```

#### `hpc_training_config.yaml` - Configuration HPC
```yaml
# Configuration optimisée pour HPC
data:
  batch_size: 32        # Batch plus large
  num_workers: 8        # Plus de workers
model:
  latent_dim: 256       # Modèle plus large
training:
  num_epochs: 200       # Plus d'époques
  use_amp: true         # Mixed precision
```

### 🚀 Scripts HPC

#### Scripts d'Installation
- **`setup_hpc.sh`**: Installation complète avec fallbacks
- **`setup_hpc_offline.sh`**: Installation hors ligne
- **`setup_asuka.sh`**: Installation spécifique Asuka
- **`setup_asuka_custom.sh`**: Installation personnalisée

#### Scripts de Vérification
- **`check_hpc_environment.sh`**: Diagnostic complet
- **`check_hpc_readiness.sh`**: Vérification préparation

#### Scripts de Transfert
- **`transfer_to_hpc.sh`**: Transfert vers HPC générique
- **`transfer_to_asuka.sh`**: Transfert vers Asuka

#### Scripts de Monitoring
- **`monitor_hpc.sh`**: Surveillance des jobs
- **`download_results.sh`**: Récupération des résultats

### 📋 Jobs SLURM

#### `train_job.slurm` - Job Standard
```bash
#!/bin/bash
#SBATCH --job-name=nlbs-cvae
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
```

#### `train_asuka.slurm` - Job Asuka
```bash
#!/bin/bash
#SBATCH --job-name=nlbs-cvae-asuka
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
```

### 🧪 Tests et Démonstrations

#### `demo.py` - Démonstration Complète
- Chargement des données
- Création du modèle
- Entraînement court
- Génération d'échantillons
- Visualisation des résultats

#### `quick_train.py` - Entraînement Rapide
- Configuration minimale
- Entraînement accéléré
- Test de fonctionnalité

#### Scripts de Test
- **`test_model.py`**: Test architecture modèle
- **`test_dataset.py`**: Test chargement données
- **`test_structure.py`**: Test structure projet
- **`test_training.py`**: Test boucle entraînement

### 📦 Gestion des Dépendances

#### `requirements.txt` - Standard
```
torch>=1.12.0
torchvision>=0.13.0
pandas>=1.3.0
pyyaml>=5.4.0
tqdm>=4.62.0
albumentations>=1.1.0
opencv-python>=4.5.0
scipy>=1.7.0
scikit-image>=0.18.0
pydicom>=2.2.0
tensorboard>=2.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

#### `requirements_hpc.txt` - HPC Optimisé
- Versions spécifiques pour HPC
- Support CUDA
- Optimisations performance

### 📁 Structure des Résultats

```
results/
├── checkpoints/          # Points de sauvegarde
│   ├── model_epoch_010.pth
│   ├── model_epoch_020.pth
│   └── best_model.pth
├── logs/                 # Logs d'entraînement
│   ├── tensorboard/
│   └── training.log
├── galleries/            # Images générées
│   ├── epoch_010/
│   ├── epoch_020/
│   └── final_samples/
└── metrics/              # Métriques d'évaluation
    ├── fid_scores.json
    ├── ssim_scores.json
    └── evaluation_report.pdf
```

## Flux de Travail

### 1. Développement Local
```bash
# Installation
pip install -r requirements.txt

# Test rapide
python quick_train.py

# Démonstration complète
python demo.py
```

### 2. Déploiement HPC
```bash
# Transfert vers HPC
./transfer_to_hpc.sh

# Installation sur HPC
./setup_hpc.sh

# Vérification
./check_hpc_environment.sh

# Soumission job
sbatch train_job.slurm
```

### 3. Monitoring et Résultats
```bash
# Surveillance
./monitor_hpc.sh

# Téléchargement résultats
./download_results.sh
```

## Caractéristiques Techniques

### Optimisations HPC
- **Mixed Precision**: Réduction mémoire GPU
- **Gradient Clipping**: Stabilité d'entraînement
- **Batch Size Adaptatif**: Selon ressources disponibles
- **Multi-GPU**: Support DataParallel
- **Checkpointing**: Sauvegarde régulière

### Robustesse
- **Fallback Installation**: Méthodes alternatives
- **Error Handling**: Gestion d'erreurs complète
- **Validation**: Tests automatiques
- **Documentation**: Guides détaillés

### Performance
- **Efficient Data Loading**: Chargement optimisé
- **Memory Management**: Gestion mémoire efficace
- **Parallel Processing**: Traitement parallèle
- **GPU Optimization**: Optimisations CUDA

## Utilisation Recommandée

1. **Développement**: Utiliser configuration locale
2. **Test**: Scripts de test pour validation
3. **Production**: Configuration HPC pour entraînement complet
4. **Monitoring**: Scripts de surveillance pour suivi
5. **Évaluation**: Scripts d'évaluation pour métriques

Cette structure modulaire permet un développement efficace, un déploiement robuste sur HPC, et une maintenance facilitée du projet NLBS-CVAE.