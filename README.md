# Docking Reward Calculator

Score molecules based on docking affinity, protein-ligand interactions, and drug-likeness metrics. Designed for high-throughput use in reinforcement learning pipelines.

**Features:**
- Two docking backends: CPU (Vina) and GPU (Uni-Dock)
- Multiple scoring functions: vina, vinardo, ad4
- Interaction-based scoring (H-bonds, hydrophobic, salt bridges, pi-stacking)
- Drug-likeness scoring (QED + custom scorers)
- Parallel processing with multiprocessing or Dask

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

### GPU Support (Uni-Dock)

For GPU-accelerated docking (100-1000x faster), install Uni-Dock:

```bash
# Install Uni-Dock
conda install -c conda-forge unidock

# Or via pip
pip install unidock
```

Then set `backend: unidock` in your config file.

### Verifying Installation

```bash
# Check that Vina is available
python -c "from vina import Vina; print('Vina OK')"

# Check that OpenBabel is available
obabel --version

# Check Uni-Dock (optional, for GPU)
which unidock

# Check that the package is installed
docking-reward --help
```

## Quick Start

### Command Line

```bash
# Basic usage (uses docking backend from config)
docking-reward \
  --input-smiles molecules.txt \
  --output-dir ./results \
  --config-file config.yaml \
  --workers 64
```

### Python API

```python
from docking_reward import RewardCalculator

# Initialize calculator (docking backend comes from config)
calc = RewardCalculator("config.yaml", n_workers=64)

# Score a list of SMILES
scores = calc.score(["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"])

# Or score from a file and save results
calc.score_file("molecules.txt", output_dir="./results")
```

## Configuration

See `example_config.yaml` for a fully commented configuration file. Key sections:

### Docking Settings
```yaml
docking:
  # Backend: "vina" (CPU) or "unidock" (GPU)
  backend: unidock

  # Scoring function: "vina", "vinardo", or "ad4"
  scoring_function: vina

  exhaustiveness: 8  # Higher = more thorough but slower
  n_poses: 9         # Number of poses to generate
  energy_range: 3.0  # Max energy difference from best pose
```

**Docking backends:**
- `vina`: CPU-based AutoDock Vina. Parallelized per-molecule using `--workers`.
- `unidock`: GPU-based Uni-Dock. Batches all ligands and docks on GPU. **100-1000x faster** for large batches.

**Scoring functions:**
- `vina`: Standard Vina scoring (most validated)
- `vinardo`: Faster Vinardo scoring (similar accuracy)
- `ad4`: AutoDock4 scoring function

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

## CLI Options

```
docking-reward --help

Options:
  -i, --input-smiles PATH   Input file with SMILES (one per line) [required]
  -o, --output-dir PATH     Output directory for results [required]
  -c, --config-file PATH    YAML configuration file [required]
  -w, --workers N           Number of parallel workers (default: 1)
  -b, --parallel-backend    "multiprocessing" or "dask" (default: multiprocessing)
  -v, --verbose             Show debug messages on console
  -q, --quiet               Suppress console output (only errors)
  --log-file PATH           Log file path (default: OUTPUT_DIR/docking.log)
```

**Console output:** Clean progress bars with summary statistics. All verbose warnings from RDKit/Vina are redirected to the log file.

## Output

```
output_dir/
├── results.csv        # Simple output: smiles, score
├── breakdown.csv      # Detailed score breakdown (see below)
├── docking.log        # Detailed log with all RDKit/Vina warnings
├── poses/             # SDF files with docked poses
│   ├── mol_0_CCO.sdf
│   └── ...
└── tmp/               # Intermediate files
    ├── ligands/       # Prepared ligand PDBQT files
    └── proteins/      # Prepared protein PDBQT files (cached)
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
