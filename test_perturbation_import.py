#!/usr/bin/env python
"""
Quick test to verify perturbation prediction code can be imported and initialized
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=" * 80)
print("Testing DeepSC Perturbation Prediction Import")
print("=" * 80)

# Test 1: Import modules
print("\n[Test 1] Importing modules...")
try:
    from deepsc.finetune.perturbation_finetune import PerturbationPredictor
    print("✓ Successfully imported PerturbationPredictor")
except Exception as e:
    print(f"✗ Failed to import PerturbationPredictor: {e}")
    sys.exit(1)

# Test 2: Check dependencies
print("\n[Test 2] Checking dependencies...")
dependencies = [
    ("torch", "PyTorch"),
    ("lightning", "Lightning Fabric"),
    ("gears", "GEARS"),
    ("hydra", "Hydra"),
    ("omegaconf", "OmegaConf"),
]

missing_deps = []
for module_name, display_name in dependencies:
    try:
        __import__(module_name)
        print(f"✓ {display_name} available")
    except ImportError:
        print(f"✗ {display_name} not found")
        missing_deps.append(display_name)

if missing_deps:
    print(f"\n⚠ Missing dependencies: {', '.join(missing_deps)}")
    print("Please install them before running perturbation prediction.")
else:
    print("\n✓ All dependencies available")

# Test 3: Check configuration files
print("\n[Test 3] Checking configuration files...")
config_files = [
    "configs/pp/pp.yaml",
    "configs/pp/model/deepsc.yaml",
]

for config_file in config_files:
    config_path = Path(__file__).parent / config_file
    if config_path.exists():
        print(f"✓ {config_file} exists")
    else:
        print(f"✗ {config_file} not found")

# Test 4: Check pretrained model path
print("\n[Test 4] Checking pretrained model...")
pretrained_path = "/home/angli/baseline/DeepSC/results/pretraining_1201/DeepSC_11_0.ckpt"
if Path(pretrained_path).exists():
    print(f"✓ Pretrained model found: {pretrained_path}")
else:
    print(f"⚠ Pretrained model not found: {pretrained_path}")
    print("  You may need to update the path in configs/pp/pp.yaml")

# Test 5: Check vocabulary CSV
print("\n[Test 5] Checking vocabulary CSV...")
csv_path = "/home/angli/baseline/DeepSC-117-t86/scripts/data/preprocessing/gene_map_tp10k.csv"
if Path(csv_path).exists():
    print(f"✓ Vocabulary CSV found: {csv_path}")
else:
    print(f"⚠ Vocabulary CSV not found: {csv_path}")
    print("  You may need to update the path in configs/pp/pp.yaml")

# Test 6: Check data directory
print("\n[Test 6] Checking data directory...")
data_path = Path("./data")
if data_path.exists():
    print(f"✓ Data directory exists: {data_path}")
    # Check if norman data exists
    norman_data = data_path / "norman"
    if norman_data.exists():
        print(f"✓ Norman dataset found")
    else:
        print(f"⚠ Norman dataset not found")
        print("  Run the following to download:")
        print("  python -c 'from gears import PertData; pert = PertData(\"./data\"); pert.load(\"norman\")'")
else:
    print(f"⚠ Data directory not found: {data_path}")
    print("  Please create it or update data_path in configs/pp/pp.yaml")

print("\n" + "=" * 80)
print("Import Test Summary")
print("=" * 80)
print("If all checks passed (✓), you can run:")
print("  python src/deepsc/finetune/run_perturbation_hydra.py")
print("\nFor more details, see:")
print("  - src/deepsc/finetune/USAGE_GUIDE.md")
print("  - src/deepsc/finetune/SUMMARY.md")
print("=" * 80)
