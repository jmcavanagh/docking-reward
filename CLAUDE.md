# Docking Reward Calculator - Development Guide

This document provides context for AI assistants working on this codebase.

## Project Overview

A Python package for scoring molecules based on docking affinity, protein-ligand interactions, and drug-likeness metrics. Designed for high-throughput use in reinforcement learning pipelines for drug discovery.

## Architecture

```
src/docking_reward/
├── calculator.py      # Main RewardCalculator orchestrator
├── config.py          # YAML config parsing and validation
├── cli.py             # Command-line interface
├── docking.py         # AutoDock Vina wrapper (CPU backend)
├── unidock.py         # Uni-Dock wrapper (GPU backend)
├── ligand_prep.py     # SMILES -> 3D -> PDBQT conversion
├── protein_prep.py    # PDB -> PDBQT conversion
├── interactions.py    # PLIP-based interaction analysis
├── druglikeness.py    # QED and custom scorers
├── parallel.py        # Multiprocessing/Dask executors
└── logging_config.py  # Logging setup with warning suppression
```

## Key Dependencies (conda-only)

These MUST be installed via conda, not pip:
- `rdkit` - Molecule handling, 3D embedding
- `vina` - AutoDock Vina docking engine
- `openbabel` - Protein PDBQT conversion (via `obabel` CLI)
- `plip` - Protein-ligand interaction profiler
- `meeko` - Ligand PDBQT preparation

## Docking Backends

1. **Vina** (`backend: vina`): CPU-based, parallelized per-molecule using multiprocessing
2. **Uni-Dock** (`backend: unidock`): GPU-based, batches all ligands for massive speedup

## Score Formula

```
total_score = Σ(target.weight × -vina_score)           # Negated: better binding = positive
            + Σ(interaction.weight × interaction_count)
            + Σ(druglikeness.weight × druglikeness_score)
```

Failed molecules return `-10.0`.

## Common Development Tasks

### Adding a new scoring component
1. Create module in `src/docking_reward/`
2. Add config parsing in `config.py`
3. Integrate in `calculator.py` (`_score_single_molecule` for Vina, `_compute_molecule_score` for Uni-Dock)
4. Update `ScoreBreakdown` dataclass
5. Add columns in `_write_breakdown_csv`

### Adding a new docking backend
1. Create wrapper module (see `unidock.py` as template)
2. Add to `VALID_DOCKING_BACKENDS` in `config.py`
3. Add scoring method in `calculator.py`
4. Update `_score_molecules` dispatch logic

### Modifying CLI
1. Edit `cli.py` - uses argparse
2. Update epilog help text
3. Update README.md CLI Options section

## Testing

```bash
pytest tests/ -v
```

Tests use fixtures to avoid needing actual PDB files for unit tests.

## Parallelization Notes

- Multiprocessing uses `spawn` context (required for RDKit/CUDA compatibility)
- Worker functions must be picklable (top-level, no lambdas)
- `ScoringContext` dataclass bundles all data sent to workers
- Uni-Dock parallelizes differently: ligand prep on CPU, docking on GPU in batch

## Logging

- Console: Clean progress bars via tqdm
- File (`docking.log`): All debug/warning messages including RDKit/Vina noise
- RDKit warnings suppressed at import via `logging_config.py`

## Config File Format

See `example_config.yaml` for fully documented example. Key sections:
- `docking:` - Backend, scoring function, search params
- `druglikeness:` - QED weight, custom scorers
- `targets:` - List of protein targets with box coordinates and interactions

## Uni-Dock Error Handling

Uni-Dock batch docking can fail if ANY ligand in the batch has issues. Common problems:
- **Unsupported atom types**: Boron ("B"), Silicon, etc. not supported by AutoDock force field
- **Too many torsions**: Default limit is 32 rotatable bonds

The code handles this automatically:
1. **Pre-filtering** (`ligand_prep.validate_pdbqt_for_unidock`): Checks atom types and torsion count before submission
2. **Retry logic** (`unidock.run_unidock_batch`): If batch fails, parses stderr to identify bad ligands, excludes them, and retries (up to 3 attempts)

Supported atom types are defined in `AUTODOCK_ATOM_TYPES` in `ligand_prep.py`.

## Common Issues

1. **"openbabel not found"**: Must install via conda, not pip
2. **Multiprocessing hangs**: Check for unpicklable objects in ScoringContext
3. **RDKit warnings flooding console**: Ensure `logging_config.setup_logging()` is called
4. **Uni-Dock slow**: Check GPU is being used (`nvidia-smi`)
5. **All Uni-Dock molecules fail**: Usually one bad ligand (unsupported atom type or too many torsions) aborting the batch - now handled with pre-filtering and retry
