import os
from pathlib import Path

artifacts_dir = Path('app/ml_artifacts')
files = sorted(list(artifacts_dir.glob('*')))

print("Artifacts in ml_artifacts directory:")
for f in files:
    size = f.stat().st_size
    print(f"  {f.name}: {size} bytes")

print("\nCredit card model files:")
cc_files = [f for f in files if 'credit_card' in f.name]
for f in cc_files:
    print(f"  ✓ {f.name}")
