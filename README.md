# Docking Reward Calculator

Score molecules based on docking affinity, protein-ligand interactions, and drug-likeness metrics. Designed for high-throughput use in reinforcement learning pipelines.

## Installation

```bash
# Using uv (recommended)
uv pip install -e .

# With Dask support for distributed computing
uv pip install -e ".[dask]"

# Note: If PLIP/OpenBabel cause issues with uv, use pip:
pip install -e .
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
├── results.csv        # smiles,score
├── poses/             # SDF files with docked poses
│   ├── mol_0_CCO.sdf
│   └── ...
└── tmp/               # Intermediate PDBQT files
    ├── ligands/
    └── proteins/
```

## License

MIT
