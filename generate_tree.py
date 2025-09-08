#!/usr/bin/env python3
"""
Générateur d'arborescence détaillée pour le projet NLBS-CVAE
Analyse tous les fichiers et génère une documentation complète
"""

import os
import sys
from pathlib import Path
import subprocess
from datetime import datetime

def get_file_info(filepath):
    """Obtient les informations détaillées d'un fichier"""
    try:
        stat = filepath.stat()
        size = stat.st_size
        
        # Taille lisible
        if size < 1024:
            size_str = f"{size} B"
        elif size < 1024 * 1024:
            size_str = f"{size/1024:.1f} KB"
        else:
            size_str = f"{size/(1024*1024):.1f} MB"
        
        # Permissions
        mode = stat.st_mode
        perms = []
        if mode & 0o400: perms.append('r')
        if mode & 0o200: perms.append('w')
        if mode & 0o100: perms.append('x')
        perm_str = ''.join(perms) if perms else '---'
        
        return {
            'size': size_str,
            'permissions': perm_str,
            'lines': count_lines(filepath) if filepath.suffix in ['.py', '.yaml', '.yml', '.sh', '.md', '.txt'] else None
        }
    except:
        return {'size': '?', 'permissions': '?', 'lines': None}

def count_lines(filepath):
    """Compte les lignes d'un fichier texte"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return sum(1 for _ in f)
    except:
        return None

def get_file_description(filepath):
    """Obtient une description du fichier basée sur son contenu"""
    descriptions = {
        # Python files
        '__init__.py': 'Module Python',
        'cvae.py': 'Conditional Variational Autoencoder - Architecture principale',
        'layers.py': 'Couches personnalisées (FiLM, ResidualBlock, Attention)',
        'losses.py': 'Fonctions de perte (VAE, Reconstruction, KL, Edge)',
        'dataset.py': 'Dataset NLBS pour mammographies avec preprocessing',
        'transforms.py': 'Transformations et augmentations d\'images',
        'utils.py': 'Fonctions utilitaires',
        'train_cvae.py': 'Script principal d\'entraînement avec validation',
        'evaluate_model.py': 'Évaluation et métriques du modèle',
        'generate_samples.py': 'Génération d\'échantillons à partir du modèle',
        'training_utils.py': 'Utilitaires pour l\'entraînement',
        'demo.py': 'Démonstration complète du pipeline',
        'quick_train.py': 'Entraînement rapide pour tests',
        'test_model.py': 'Tests de l\'architecture du modèle',
        'test_dataset.py': 'Tests du chargement des données',
        'test_structure.py': 'Tests de la structure du projet',
        'test_training.py': 'Tests de la boucle d\'entraînement',
        
        # Config files
        'training_config.yaml': 'Configuration d\'entraînement locale',
        'hpc_training_config.yaml': 'Configuration optimisée pour HPC',
        'requirements.yaml': 'Dépendances Conda',
        
        # Shell scripts
        'setup_hpc.sh': 'Installation principale sur HPC avec fallbacks',
        'setup_hpc_offline.sh': 'Installation hors ligne pour HPC',
        'setup_asuka.sh': 'Installation spécifique pour Asuka HPC',
        'setup_asuka_custom.sh': 'Installation personnalisée Asuka',
        'check_hpc_environment.sh': 'Diagnostic complet de l\'environnement HPC',
        'check_hpc_readiness.sh': 'Vérification de préparation HPC',
        'transfer_to_hpc.sh': 'Transfert de fichiers vers HPC',
        'transfer_to_asuka.sh': 'Transfert vers Asuka HPC',
        'download_results.sh': 'Téléchargement des résultats depuis HPC',
        'monitor_hpc.sh': 'Surveillance des jobs HPC',
        
        # SLURM files
        'train_job.slurm': 'Job SLURM pour entraînement standard',
        'train_asuka.slurm': 'Job SLURM optimisé pour Asuka',
        
        # Documentation
        'README.md': 'Documentation principale du projet',
        'NLBS_CVAE_Architecture_Guide.md': 'Guide détaillé de l\'architecture',
        'HPC_INSTALLATION_GUIDE.md': 'Guide d\'installation sur HPC',
        'ASUKA_TRANSFER_GUIDE.md': 'Guide de transfert vers Asuka',
        'ASUKA_QUICKSTART.md': 'Démarrage rapide sur Asuka',
        'hpc_setup.md': 'Configuration HPC',
        'PROJECT_STRUCTURE.md': 'Structure détaillée du projet',
        
        # Requirements
        'requirements.txt': 'Dépendances Python standard',
        'requirements_hpc.txt': 'Dépendances optimisées pour HPC',
    }
    
    filename = filepath.name
    if filename in descriptions:
        return descriptions[filename]
    
    # Description basée sur l'extension
    ext_descriptions = {
        '.py': 'Script Python',
        '.yaml': 'Fichier de configuration YAML',
        '.yml': 'Fichier de configuration YAML',
        '.sh': 'Script shell',
        '.slurm': 'Script de job SLURM',
        '.md': 'Documentation Markdown',
        '.txt': 'Fichier texte',
        '.json': 'Fichier JSON',
    }
    
    return ext_descriptions.get(filepath.suffix, 'Fichier')

def generate_tree(root_path, exclude_dirs=None, exclude_files=None):
    """Génère l'arborescence détaillée"""
    if exclude_dirs is None:
        exclude_dirs = {'.venv', '__pycache__', '.git', 'nlbs-cvae-env', '.pytest_cache'}
    if exclude_files is None:
        exclude_files = {'*.pyc', '*.pyo', '*.pyd', '.DS_Store'}
    
    root = Path(root_path)
    tree_lines = []
    
    def should_exclude(path):
        if path.is_dir():
            return path.name in exclude_dirs
        return any(path.match(pattern) for pattern in exclude_files)
    
    def add_directory(dir_path, prefix="", is_last=True):
        if should_exclude(dir_path):
            return
        
        # Nom du dossier avec icône
        folder_icon = "📁"
        if dir_path.name in ['models', 'training']:
            folder_icon = "🧠"
        elif dir_path.name in ['data']:
            folder_icon = "📊"
        elif dir_path.name in ['configs']:
            folder_icon = "⚙️"
        elif dir_path.name in ['results']:
            folder_icon = "📈"
        elif dir_path.name in ['utils']:
            folder_icon = "🔧"
        elif dir_path.name in ['assessment']:
            folder_icon = "📋"
        
        connector = "└── " if is_last else "├── "
        tree_lines.append(f"{prefix}{connector}{folder_icon} {dir_path.name}/")
        
        # Nouveau préfixe pour les enfants
        child_prefix = prefix + ("    " if is_last else "│   ")
        
        # Lister les contenus
        try:
            contents = sorted([p for p in dir_path.iterdir() if not should_exclude(p)],
                            key=lambda x: (x.is_file(), x.name.lower()))
            
            for i, item in enumerate(contents):
                is_last_item = (i == len(contents) - 1)
                
                if item.is_dir():
                    add_directory(item, child_prefix, is_last_item)
                else:
                    add_file(item, child_prefix, is_last_item)
        except PermissionError:
            tree_lines.append(f"{child_prefix}[Permission denied]")
    
    def add_file(file_path, prefix="", is_last=True):
        if should_exclude(file_path):
            return
        
        # Icône basée sur l'extension
        file_icon = "📄"
        if file_path.suffix == '.py':
            file_icon = "🐍"
        elif file_path.suffix in ['.yaml', '.yml']:
            file_icon = "⚙️"
        elif file_path.suffix == '.sh':
            file_icon = "🔧"
        elif file_path.suffix == '.slurm':
            file_icon = "🚀"
        elif file_path.suffix == '.md':
            file_icon = "📖"
        elif file_path.suffix == '.txt':
            file_icon = "📝"
        elif file_path.name.startswith('requirements'):
            file_icon = "📦"
        
        connector = "└── " if is_last else "├── "
        
        # Informations du fichier
        info = get_file_info(file_path)
        description = get_file_description(file_path)
        
        # Ligne principale
        line = f"{prefix}{connector}{file_icon} {file_path.name}"
        
        # Ajout des détails
        details = []
        if info['size'] != '?':
            details.append(f"{info['size']}")
        if info['lines'] is not None:
            details.append(f"{info['lines']} lignes")
        if info['permissions'] != '?':
            details.append(f"[{info['permissions']}]")
        
        if details:
            line += f" ({', '.join(details)})"
        
        tree_lines.append(line)
        
        # Description sur la ligne suivante
        if description and description != 'Fichier':
            desc_prefix = prefix + ("    " if is_last else "│   ")
            tree_lines.append(f"{desc_prefix}    ↳ {description}")
    
    # Commencer par le dossier racine
    tree_lines.append(f"📁 {root.name}/")
    
    # Traiter le contenu
    try:
        contents = sorted([p for p in root.iterdir() if not should_exclude(p)],
                        key=lambda x: (x.is_file(), x.name.lower()))
        
        for i, item in enumerate(contents):
            is_last_item = (i == len(contents) - 1)
            
            if item.is_dir():
                add_directory(item, "", is_last_item)
            else:
                add_file(item, "", is_last_item)
    except PermissionError:
        tree_lines.append("[Permission denied]")
    
    return tree_lines

def get_project_stats(root_path):
    """Calcule les statistiques du projet"""
    root = Path(root_path)
    stats = {
        'total_files': 0,
        'python_files': 0,
        'config_files': 0,
        'script_files': 0,
        'doc_files': 0,
        'total_lines': 0,
        'python_lines': 0,
        'total_size': 0
    }
    
    exclude_dirs = {'.venv', '__pycache__', '.git', 'nlbs-cvae-env'}
    
    for file_path in root.rglob('*'):
        if file_path.is_file() and not any(excl in file_path.parts for excl in exclude_dirs):
            stats['total_files'] += 1
            
            try:
                size = file_path.stat().st_size
                stats['total_size'] += size
                
                if file_path.suffix == '.py':
                    stats['python_files'] += 1
                    lines = count_lines(file_path)
                    if lines:
                        stats['python_lines'] += lines
                        stats['total_lines'] += lines
                elif file_path.suffix in ['.yaml', '.yml']:
                    stats['config_files'] += 1
                    lines = count_lines(file_path)
                    if lines:
                        stats['total_lines'] += lines
                elif file_path.suffix in ['.sh', '.slurm']:
                    stats['script_files'] += 1
                    lines = count_lines(file_path)
                    if lines:
                        stats['total_lines'] += lines
                elif file_path.suffix in ['.md', '.txt']:
                    stats['doc_files'] += 1
                    lines = count_lines(file_path)
                    if lines:
                        stats['total_lines'] += lines
            except:
                pass
    
    return stats

def main():
    """Fonction principale"""
    root_path = Path(__file__).parent
    
    print("🌳 Génération de l'arborescence détaillée du projet NLBS-CVAE")
    print("=" * 70)
    
    # Générer l'arborescence
    tree_lines = generate_tree(root_path)
    
    # Calculer les statistiques
    stats = get_project_stats(root_path)
    
    # Créer le contenu complet
    content = []
    content.append("# NLBS-CVAE - Arborescence Détaillée du Projet")
    content.append("")
    content.append(f"**Généré le:** {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}")
    content.append("")
    
    # Statistiques
    content.append("## 📊 Statistiques du Projet")
    content.append("")
    content.append(f"- **Fichiers totaux:** {stats['total_files']}")
    content.append(f"- **Fichiers Python:** {stats['python_files']} ({stats['python_lines']} lignes)")
    content.append(f"- **Fichiers de configuration:** {stats['config_files']}")
    content.append(f"- **Scripts:** {stats['script_files']}")
    content.append(f"- **Documentation:** {stats['doc_files']}")
    content.append(f"- **Lignes totales:** {stats['total_lines']}")
    
    size_mb = stats['total_size'] / (1024 * 1024)
    content.append(f"- **Taille totale:** {size_mb:.1f} MB")
    content.append("")
    
    # Arborescence
    content.append("## 🌳 Structure Complète")
    content.append("")
    content.append("```")
    content.extend(tree_lines)
    content.append("```")
    content.append("")
    
    # Légende
    content.append("## 📋 Légende")
    content.append("")
    content.append("### Icônes")
    content.append("- 📁 Dossier")
    content.append("- 🧠 Modèles et entraînement")
    content.append("- 📊 Données")
    content.append("- ⚙️ Configuration")
    content.append("- 📈 Résultats")
    content.append("- 🔧 Utilitaires")
    content.append("- 🐍 Fichier Python")
    content.append("- 🚀 Script SLURM")
    content.append("- 📖 Documentation")
    content.append("- 📦 Dépendances")
    content.append("")
    
    content.append("### Informations")
    content.append("- **(taille, lignes, permissions)** - Détails du fichier")
    content.append("- **↳ Description** - Fonction du fichier")
    content.append("")
    
    # Sauvegarder
    output_file = root_path / "DETAILED_TREE.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(content))
    
    print(f"✅ Arborescence générée dans: {output_file}")
    print(f"📊 {stats['total_files']} fichiers analysés")
    print(f"📝 {stats['total_lines']} lignes de code")
    
    # Afficher un aperçu
    print("\n🌳 Aperçu de l'arborescence:")
    print("-" * 50)
    for line in tree_lines[:20]:  # Premiers 20 éléments
        print(line)
    if len(tree_lines) > 20:
        print("...")
        print(f"[{len(tree_lines) - 20} éléments supplémentaires]")

if __name__ == "__main__":
    main()