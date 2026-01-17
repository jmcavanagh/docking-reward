# Docking Reward Calculator

Score molecules based on docking affinity, protein-ligand interactions, and drug-likeness metrics. Designed for high-throughput use in reinforcement learning pipelines.

## Installation

This package requires several tools with native dependencies (AutoDock Vina, OpenBabel, PLIP). **Conda is required** for installation.

```bash
# Create a new conda environment (recommended)
conda create -n docking-reward python=3.11
conda activate docking-reward

# Install native dependencies from conda-forge
conda install -c conda-forge vina openbabel plip rdkit meeko numpy pyyaml

# Install this package
pip install -e .

# With Dask support for distributed computing
pip install -e ".[dask]"
```

### Verifying Installation

```bash
# Check that Vina is available
python -c "from vina import Vina; print('Vina OK')"

# Check that OpenBabel is available
obabel --version

# Check that the package is installed
docking-reward --help
```

## Quick Start

### Command Line

```bash
docking-reward \
  --input-smiles molecules.txt \
  --output-dir ./results \
  --config-file config.yaml \
  --workers 16 \
  --backend multiprocessing
```

### Python API

```python
from docking_reward import RewardCalculator

# Initialize calculator
calc = RewardCalculator("config.yaml", n_workers=8)

# Score a list of SMILES
scores = calc.score(["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"])

# Or score from a file and save results
calc.score_file("molecules.txt", output_dir="./results")
```

## Configuration

See `example_config.yaml` for a fully commented configuration file. Key sections:

### Vina Settings
```yaml
vina:
  exhaustiveness: 8  # Higher = more thorough but slower
  n_poses: 9         # Number of poses to generate
  energy_range: 3.0  # Max energy difference from best pose
```

### Drug-likeness
```yaml
druglikeness:
  qed:
    weight: 1.0  # QED score contribution
  # Custom scorers can be added:
  # custom:
  #   - path: "/path/to/scorer.py"
  #     function: "score"
  #     weight: 0.5
```

### Docking Targets
```yaml
targets:
  - name: "main_target"
    pdb_path: "/path/to/protein.pdb"
    center: [15.0, 20.0, 25.0]  # Box center (x, y, z) in Angstroms
    size: [20.0, 20.0, 20.0]    # Box size (x, y, z) in Angstroms
    weight: 1.0                  # Positive = reward good binding

    # Optional: interaction-based scoring
    interactions:
      - residue: "ASP123"
        type: "hydrogen_bond"  # or hydrophobic, salt_bridge, pi_stacking, any
        weight: 0.5            # Score += weight * count
```

## Score Formula

```
total_score = Σ(target.weight × -vina_score)           # Negated so positive = better
            + Σ(interaction.weight × interaction_count)
            + Σ(druglikeness.weight × druglikeness_score)
```

Invalid SMILES or failed docking returns `-10.0`.

## Output

```
output_dir/
├── results.csv        # Simple output: smiles, score
├── breakdown.csv      # Detailed score breakdown (see below)
└── poses/             # SDF files with docked poses
    ├── mol_0_CCO.sdf
    └── ...
```

Temporary files are stored in `./tmp/` (configurable):
```
tmp/
├── ligands/           # Prepared ligand PDBQT files
└── proteins/          # Prepared protein PDBQT files (cached)
```

### breakdown.csv Columns

The `breakdown.csv` file provides a detailed breakdown of all score components for debugging and analysis:

| Column | Description |
|--------|-------------|
| `smiles` | Input SMILES string |
| `total_score` | Final combined score |
| `error` | Error message if scoring failed (empty if successful) |

**Per-target columns** (repeated for each target defined in config):

| Column | Description |
|--------|-------------|
| `{target}_vina_score` | Raw Vina docking score in kcal/mol (negative = better binding). Empty if docking failed. |
| `{target}_docking_contribution` | Contribution to total score: `weight × (-vina_score)`. Positive weight makes better binding increase the score. |
| `{target}_interaction_score` | Sum of interaction scores for this target: `Σ(interaction_weight × count)`. Only non-zero if interactions are configured. |
| `{target}_interaction_pose` | Index (0-based) of the pose used for interaction scoring. This is the pose within the energy threshold that had the highest interaction score. |

**Drug-likeness columns**:

| Column | Description |
|--------|-------------|
| `qed_score` | QED (Quantitative Estimate of Drug-likeness) value, 0-1 scale. Empty if QED not configured. |
| `qed_contribution` | Contribution to total score: `qed_weight × qed_score`. |

### Example breakdown.csv

For a config with targets "pgk2" (main, weight=1.0) and "off_target" (anti-target, weight=-0.8):

```csv
smiles,total_score,error,pgk2_vina_score,pgk2_docking_contribution,pgk2_interaction_score,pgk2_interaction_pose,off_target_vina_score,off_target_docking_contribution,off_target_interaction_score,off_target_interaction_pose,qed_score,qed_contribution
CCO,3.2500,,−5.2,5.2000,0.0000,0,−4.5,−3.6000,0.0000,0,0.4500,0.4500
c1ccccc1,-10.0000,Docking failed: pgk2: No poses found,,0.0000,0.0000,,,0.0000,0.0000,,,
```

## License

MIT
