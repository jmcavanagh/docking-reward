# Docking Reward Calculator - Technical Specification

## Overview

A Python package for scoring molecules (given as SMILES strings) based on:
1. Docking affinity to one or more protein targets (via AutoDock Vina)
2. Penalized/rewarded binding to anti-targets (negative weights)
3. Specific protein-ligand interactions (H-bonds, hydrophobic, etc.)
4. Drug-likeness metrics (QED, with plugin support for custom scorers)

Designed for high-throughput use in reinforcement learning pipelines, with support for up to 64+ parallel workers.

---

## Project Structure

```
docking-reward/
├── pyproject.toml                 # Package metadata, dependencies, entry points
├── README.md                      # User documentation
├── example_config.yaml            # Heavily commented example configuration
├── src/
│   └── docking_reward/
│       ├── __init__.py            # Public API exports
│       ├── cli.py                 # Command-line interface
│       ├── config.py              # Configuration parsing and validation
│       ├── calculator.py          # Main RewardCalculator orchestrator
│       ├── docking.py             # Vina docking wrapper
│       ├── ligand_prep.py         # SMILES → 3D conformer → PDBQT
│       ├── protein_prep.py        # PDB → PDBQT conversion (Meeko)
│       ├── interactions.py        # PLIP-based interaction detection
│       ├── druglikeness.py        # QED and custom scorer plugin system
│       └── parallel.py            # Multiprocessing and Dask executors
└── tests/
    ├── test_config.py
    ├── test_ligand_prep.py
    ├── test_docking.py
    ├── test_interactions.py
    ├── test_druglikeness.py
    └── test_calculator.py
```

---

## Dependencies

### Core Dependencies
- `python >= 3.10`
- `rdkit` - Molecule handling, 3D embedding, QED calculation
- `meeko` - Ligand and protein PDBQT preparation
- `vina` - Python bindings for AutoDock Vina
- `plip` - Protein-Ligand Interaction Profiler
- `openbabel` - Required by PLIP (may need conda/system install)
- `pyyaml` - Configuration parsing
- `numpy` - Numerical operations
- `dask[distributed]` - Optional, for Dask-based parallelism

### Dev Dependencies
- `pytest`
- `pytest-cov`

---

## Configuration File Format (YAML)

```yaml
# =============================================================================
# Docking Reward Calculator Configuration
# =============================================================================

# -----------------------------------------------------------------------------
# Global Vina Settings
# These apply to all docking runs unless overridden per-target
# -----------------------------------------------------------------------------
vina:
  # Number of Monte Carlo runs. Higher = more thorough but slower.
  # Recommended: 8 (default), increase to 16-32 for production runs
  exhaustiveness: 8

  # Number of binding poses to generate per ligand
  n_poses: 9

  # Maximum energy difference (kcal/mol) between best and worst pose
  energy_range: 3.0

  # Random seed for reproducibility (optional, omit for random)
  # seed: 42

# -----------------------------------------------------------------------------
# Drug-likeness Scoring
# Each scorer contributes: weight * score to the final reward
# -----------------------------------------------------------------------------
druglikeness:
  # Quantitative Estimate of Drug-likeness (0 to 1, higher = more drug-like)
  qed:
    weight: 1.0

  # Custom scorers: specify path to Python file with score(mol) -> float function
  # custom:
  #   - path: "/path/to/my_scorer.py"
  #     function: "score"  # function name in the file
  #     weight: 0.5

# -----------------------------------------------------------------------------
# Docking Targets
# Each target is a protein to dock against
# -----------------------------------------------------------------------------
targets:
  # ---- Example: Primary target (want strong binding) ----
  - name: "main_target"
    # Path to protein structure file (PDB format)
    # Will be automatically converted to PDBQT
    pdb_path: "/path/to/target.pdb"

    # Docking box center coordinates (x, y, z) in Angstroms
    # Typically centered on the binding site
    center: [15.0, 20.0, 25.0]

    # Docking box dimensions (x, y, z) in Angstroms
    # Should be large enough to encompass the binding site + some buffer
    # Typical values: 15-25 Angstroms per dimension
    size: [20.0, 20.0, 20.0]

    # Scoring weight for this target's docking score
    # Positive weight: better binding (more negative Vina score) = higher reward
    # Negative weight: better binding = lower reward (use for anti-targets)
    #
    # The Vina score is negated internally, so:
    #   - Vina score of -8.0 kcal/mol with weight +1.0 contributes +8.0 to reward
    #   - Vina score of -8.0 kcal/mol with weight -1.0 contributes -8.0 to reward
    weight: 1.0

    # Optional: per-target Vina settings (override global)
    # exhaustiveness: 16

    # ---- Interaction-based scoring (optional) ----
    # Award/penalize specific interactions with residues
    # Score contribution: weight * count (number of interactions of that type)
    interactions:
      # Reward hydrogen bonds with ASP123
      - residue: "ASP123"
        # Interaction type: hydrogen_bond, hydrophobic, salt_bridge, pi_stacking, any
        type: "hydrogen_bond"
        # Each H-bond with ASP123 adds 0.5 to the score
        weight: 0.5

      # Penalize any contact with TYR456
      - residue: "TYR456"
        type: "any"
        weight: -0.3

  # ---- Example: Anti-target (want weak binding) ----
  - name: "off_target"
    pdb_path: "/path/to/antitarget.pdb"
    center: [10.0, 10.0, 10.0]
    size: [18.0, 18.0, 18.0]
    # Negative weight: penalize good binding
    weight: -0.5
```

---

## Module Specifications

### 1. `config.py` - Configuration Management

#### Classes

```python
@dataclass
class VinaConfig:
    """Global Vina docking parameters."""
    exhaustiveness: int = 8
    n_poses: int = 9
    energy_range: float = 3.0
    seed: Optional[int] = None

@dataclass
class InteractionConfig:
    """Configuration for a single residue interaction scorer."""
    residue: str           # e.g., "ASP123"
    type: str              # hydrogen_bond, hydrophobic, salt_bridge, pi_stacking, any
    weight: float          # Score contribution per interaction count

@dataclass
class TargetConfig:
    """Configuration for a single docking target."""
    name: str
    pdb_path: Path
    center: Tuple[float, float, float]
    size: Tuple[float, float, float]
    weight: float
    interactions: List[InteractionConfig] = field(default_factory=list)
    # Per-target Vina overrides (optional)
    exhaustiveness: Optional[int] = None
    n_poses: Optional[int] = None
    energy_range: Optional[float] = None

@dataclass
class QEDConfig:
    """QED drug-likeness scorer config."""
    weight: float = 1.0

@dataclass
class CustomScorerConfig:
    """Custom drug-likeness scorer config."""
    path: Path
    function: str
    weight: float

@dataclass
class DruglikenessConfig:
    """All drug-likeness scoring configuration."""
    qed: Optional[QEDConfig] = None
    custom: List[CustomScorerConfig] = field(default_factory=list)

@dataclass
class Config:
    """Root configuration object."""
    vina: VinaConfig
    druglikeness: DruglikenessConfig
    targets: List[TargetConfig]
```

#### Functions

```python
def load_config(path: Path) -> Config:
    """
    Load and validate configuration from YAML file.

    Args:
        path: Path to YAML configuration file

    Returns:
        Validated Config object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid (missing required fields, invalid paths, etc.)
    """

def validate_config(config: Config) -> None:
    """
    Validate configuration for semantic correctness.

    Checks:
    - All PDB paths exist and are readable
    - Box dimensions are positive
    - Interaction types are valid
    - Custom scorer paths exist
    - At least one target is defined

    Raises:
        ValueError: With descriptive message if validation fails
    """
```

---

### 2. `ligand_prep.py` - Ligand Preparation

#### Functions

```python
def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    """
    Parse SMILES string to RDKit Mol object.

    Args:
        smiles: SMILES string

    Returns:
        RDKit Mol object, or None if parsing fails
    """

def embed_molecule(mol: Chem.Mol, n_conformers: int = 1) -> Optional[Chem.Mol]:
    """
    Generate 3D conformer(s) for a molecule using ETKDG.

    Args:
        mol: RDKit Mol object (2D)
        n_conformers: Number of conformers to generate

    Returns:
        Mol with 3D coordinates, or None if embedding fails

    Notes:
        - Adds hydrogens appropriate for pH 7.4
        - Uses ETKDG (Experimental-Torsion Knowledge Distance Geometry)
        - Performs force-field optimization (MMFF94)
    """

def mol_to_pdbqt(mol: Chem.Mol, output_path: Path) -> bool:
    """
    Convert RDKit Mol to PDBQT format using Meeko.

    Args:
        mol: RDKit Mol with 3D coordinates
        output_path: Where to write the PDBQT file

    Returns:
        True if successful, False otherwise
    """

def prepare_ligand(smiles: str, output_dir: Path) -> Optional[Path]:
    """
    Full ligand preparation pipeline: SMILES → 3D → PDBQT.

    Args:
        smiles: Input SMILES string
        output_dir: Directory for output files

    Returns:
        Path to PDBQT file, or None if any step fails
    """
```

---

### 3. `protein_prep.py` - Protein Preparation

#### Functions

```python
def pdb_to_pdbqt(pdb_path: Path, output_path: Path) -> bool:
    """
    Convert PDB file to PDBQT format using Meeko.

    Args:
        pdb_path: Input PDB file path
        output_path: Output PDBQT file path

    Returns:
        True if successful, False otherwise

    Notes:
        - Removes water molecules
        - Adds polar hydrogens
        - Computes Gasteiger charges
    """

def prepare_proteins(targets: List[TargetConfig], cache_dir: Path) -> Dict[str, Path]:
    """
    Prepare all protein targets, with caching.

    Args:
        targets: List of target configurations
        cache_dir: Directory to store prepared PDBQT files

    Returns:
        Dict mapping target name to PDBQT path

    Notes:
        - Skips preparation if PDBQT already exists and is newer than PDB
        - Thread-safe for parallel access
    """
```

---

### 4. `docking.py` - Vina Docking

#### Classes

```python
@dataclass
class DockingResult:
    """Result of a single docking run."""
    target_name: str
    scores: List[float]       # Vina scores for each pose (kcal/mol, negative)
    poses: List[Chem.Mol]     # RDKit Mol objects for each pose
    best_score: float         # Best (most negative) Vina score
    success: bool             # Whether docking succeeded
    error: Optional[str]      # Error message if failed
```

#### Functions

```python
def dock_ligand(
    ligand_pdbqt: Path,
    protein_pdbqt: Path,
    center: Tuple[float, float, float],
    size: Tuple[float, float, float],
    exhaustiveness: int = 8,
    n_poses: int = 9,
    energy_range: float = 3.0,
    seed: Optional[int] = None
) -> DockingResult:
    """
    Dock a ligand to a protein using Vina.

    Args:
        ligand_pdbqt: Path to prepared ligand PDBQT
        protein_pdbqt: Path to prepared protein PDBQT
        center: Docking box center (x, y, z) in Angstroms
        size: Docking box size (x, y, z) in Angstroms
        exhaustiveness: Vina exhaustiveness parameter
        n_poses: Number of poses to return
        energy_range: Max energy difference from best pose
        seed: Random seed for reproducibility

    Returns:
        DockingResult with scores and poses
    """

def dock_to_targets(
    ligand_pdbqt: Path,
    targets: List[TargetConfig],
    protein_pdbqts: Dict[str, Path],
    global_vina_config: VinaConfig
) -> Dict[str, DockingResult]:
    """
    Dock a ligand to all configured targets.

    Args:
        ligand_pdbqt: Path to prepared ligand
        targets: List of target configurations
        protein_pdbqts: Dict mapping target name to PDBQT path
        global_vina_config: Global Vina settings

    Returns:
        Dict mapping target name to DockingResult
    """
```

---

### 5. `interactions.py` - Interaction Analysis

#### Classes

```python
@dataclass
class InteractionCount:
    """Counts of different interaction types with a residue."""
    residue: str
    hydrogen_bonds: int = 0
    hydrophobic: int = 0
    salt_bridges: int = 0
    pi_stacking: int = 0

    @property
    def any(self) -> int:
        """Total count of all interactions."""
        return self.hydrogen_bonds + self.hydrophobic + self.salt_bridges + self.pi_stacking
```

#### Functions

```python
def analyze_interactions(
    ligand_mol: Chem.Mol,
    protein_pdb: Path
) -> List[InteractionCount]:
    """
    Analyze protein-ligand interactions using PLIP.

    Args:
        ligand_mol: RDKit Mol of docked pose
        protein_pdb: Path to protein PDB file

    Returns:
        List of InteractionCount objects, one per interacting residue

    Notes:
        - Creates temporary complex PDB for PLIP analysis
        - Parses PLIP output to extract interaction counts
    """

def score_interactions(
    interaction_counts: List[InteractionCount],
    interaction_configs: List[InteractionConfig]
) -> float:
    """
    Calculate interaction-based score contribution.

    Args:
        interaction_counts: Detected interactions from PLIP
        interaction_configs: Configured interaction weights

    Returns:
        Total interaction score (sum of weight * count for each configured interaction)

    Example:
        If ASP123 has 2 hydrogen bonds and config specifies weight=0.5 for H-bonds,
        contribution is 2 * 0.5 = 1.0
    """
```

---

### 6. `druglikeness.py` - Drug-likeness Scoring

#### Functions

```python
def calculate_qed(mol: Chem.Mol) -> float:
    """
    Calculate QED (Quantitative Estimate of Drug-likeness).

    Args:
        mol: RDKit Mol object

    Returns:
        QED score between 0 and 1 (higher = more drug-like)
    """

def load_custom_scorer(path: Path, function_name: str) -> Callable[[Chem.Mol], float]:
    """
    Dynamically load a custom scoring function from a Python file.

    Args:
        path: Path to Python file
        function_name: Name of function to load

    Returns:
        Callable that takes an RDKit Mol and returns a float score

    Raises:
        ImportError: If file can't be loaded
        AttributeError: If function doesn't exist in file
    """

def calculate_druglikeness(
    mol: Chem.Mol,
    config: DruglikenessConfig
) -> float:
    """
    Calculate total drug-likeness score from all configured scorers.

    Args:
        mol: RDKit Mol object
        config: Drug-likeness configuration

    Returns:
        Weighted sum of all drug-likeness scores
    """
```

---

### 7. `parallel.py` - Parallel Execution

#### Classes

```python
class Executor(ABC):
    """Abstract base class for parallel executors."""

    @abstractmethod
    def map(self, func: Callable, items: List[Any]) -> List[Any]:
        """Apply function to all items in parallel."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Clean up executor resources."""
        pass

class MultiprocessingExecutor(Executor):
    """Executor using Python's multiprocessing.Pool."""

    def __init__(self, n_workers: int):
        """
        Args:
            n_workers: Number of parallel workers
        """

class DaskExecutor(Executor):
    """Executor using Dask for distributed computing."""

    def __init__(self, n_workers: int, threads_per_worker: int = 1):
        """
        Args:
            n_workers: Number of Dask workers
            threads_per_worker: Threads per worker (default 1 for CPU-bound docking)
        """
```

#### Functions

```python
def get_executor(backend: str, n_workers: int) -> Executor:
    """
    Factory function to create the appropriate executor.

    Args:
        backend: "multiprocessing" or "dask"
        n_workers: Number of parallel workers

    Returns:
        Configured Executor instance
    """
```

---

### 8. `calculator.py` - Main Orchestrator

#### Classes

```python
class RewardCalculator:
    """
    Main class for calculating docking-based reward scores.

    Orchestrates ligand preparation, docking, interaction analysis,
    and drug-likeness scoring with parallel execution.
    """

    def __init__(
        self,
        config_path: Union[str, Path],
        n_workers: int = 1,
        backend: str = "multiprocessing",
        temp_dir: Optional[Path] = None
    ):
        """
        Initialize the reward calculator.

        Args:
            config_path: Path to YAML configuration file
            n_workers: Number of parallel workers
            backend: "multiprocessing" or "dask"
            temp_dir: Directory for temporary files (default: creates tmp/ subdir)
        """

    def score(self, smiles_list: List[str]) -> List[float]:
        """
        Score a list of SMILES strings.

        Args:
            smiles_list: List of SMILES strings to score

        Returns:
            List of reward scores (same order as input)
            Invalid SMILES or failed docking returns -10.0
        """

    def score_file(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path]
    ) -> Path:
        """
        Score SMILES from a file and write results.

        Args:
            input_path: Path to file with one SMILES per line
            output_dir: Directory for output files

        Returns:
            Path to results.csv

        Creates:
            output_dir/
            ├── results.csv           # SMILES,score
            └── poses/
                ├── mol_0_<smiles>.sdf
                ├── mol_1_<smiles>.sdf
                └── ...
        """

    def _score_single(self, smiles: str, index: int, output_dir: Optional[Path]) -> Tuple[float, Optional[Path]]:
        """
        Score a single SMILES string (internal method, runs in worker).

        Args:
            smiles: SMILES string
            index: Index for output file naming
            output_dir: If provided, save poses here

        Returns:
            Tuple of (score, path_to_sdf_or_none)
        """
```

#### Score Calculation Formula

For each valid SMILES that docks successfully:

```
total_score = 0

# Docking scores (negated so better binding = positive)
for target in targets:
    vina_score = dock(ligand, target)  # e.g., -8.5
    total_score += target.weight * (-vina_score)  # e.g., 1.0 * 8.5 = 8.5

    # Interaction scores
    interactions = analyze(docked_pose, target.protein)
    for interaction_config in target.interactions:
        count = get_count(interactions, interaction_config.residue, interaction_config.type)
        total_score += interaction_config.weight * count

# Drug-likeness scores
total_score += qed_weight * calculate_qed(mol)
for custom_scorer in custom_scorers:
    total_score += custom_scorer.weight * custom_scorer.score(mol)

return total_score
```

---

### 9. `cli.py` - Command-Line Interface

#### Entry Point

```python
def main():
    """
    CLI entry point: docking-reward

    Usage:
        docking-reward --input-smiles <file> --output-dir <dir> --config-file <yaml> [options]

    Arguments:
        --input-smiles, -i   Path to file with SMILES (one per line) [required]
        --output-dir, -o     Directory for results and poses [required]
        --config-file, -c    Path to YAML configuration [required]
        --workers, -w        Number of parallel workers [default: 1]
        --backend, -b        Parallelization backend: multiprocessing or dask [default: multiprocessing]
        --verbose, -v        Enable verbose logging
        --help, -h           Show help message

    Output:
        Creates output_dir with:
        - results.csv: Two columns (smiles, score)
        - poses/: Directory with SDF files for each valid molecule
        - tmp/: Intermediate PDBQT files (kept for debugging)

    Exit codes:
        0: Success
        1: Configuration error
        2: Input file error
        3: Runtime error
    """
```

---

### 10. `__init__.py` - Public API

```python
"""
Docking Reward Calculator

A tool for scoring molecules based on docking, interactions, and drug-likeness.

Example usage:
    from docking_reward import RewardCalculator

    calc = RewardCalculator("config.yaml", n_workers=8)
    scores = calc.score(["CCO", "c1ccccc1"])
"""

from .calculator import RewardCalculator
from .config import (
    Config,
    VinaConfig,
    TargetConfig,
    InteractionConfig,
    DruglikenessConfig,
    load_config,
)

__version__ = "0.1.0"
__all__ = [
    "RewardCalculator",
    "Config",
    "VinaConfig",
    "TargetConfig",
    "InteractionConfig",
    "DruglikenessConfig",
    "load_config",
]
```

---

## Error Handling

### Sentinel Values
- **Invalid SMILES**: Returns score of `-10.0`
- **Failed embedding**: Returns score of `-10.0`
- **Failed docking**: Returns score of `-10.0`
- **Failed interaction analysis**: Logs warning, contribution is `0.0`

### Logging
- Uses Python's `logging` module
- Default level: `WARNING`
- With `--verbose`: `INFO`
- Structured log format: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`

---

## File Naming Conventions

### Pose Files
Format: `mol_{index}_{sanitized_smiles}.sdf`

Where `sanitized_smiles` is the SMILES string with special characters replaced:
- `/` → `_slash_`
- `\` → `_backslash_`
- `#` → `_hash_`
- Truncated to 50 characters if longer

Example: `mol_0_CCO.sdf`, `mol_5_c1ccc_slash_C_backslash_cc1.sdf`

### Temporary Files
- Ligand PDBQTs: `tmp/ligands/mol_{index}.pdbqt`
- Protein PDBQTs: `tmp/proteins/{target_name}.pdbqt`

---

## Performance Considerations

1. **Protein preparation**: Done once at initialization, cached in `tmp/proteins/`
2. **Parallelization granularity**: One molecule per worker task
3. **Memory**: Each worker loads its own copy of protein structures
4. **Dask**: Uses `threads_per_worker=1` since Vina is CPU-bound
5. **Batch size**: For very large inputs with Dask, consider chunking to avoid scheduler overhead

---

## Testing Strategy

### Unit Tests
- `test_config.py`: YAML parsing, validation, edge cases
- `test_ligand_prep.py`: SMILES parsing, embedding, PDBQT generation
- `test_protein_prep.py`: PDB to PDBQT conversion
- `test_docking.py`: Mock Vina calls, result parsing
- `test_interactions.py`: PLIP parsing, score calculation
- `test_druglikeness.py`: QED calculation, custom loader

### Integration Tests
- End-to-end test with small molecule set
- Test with intentionally invalid SMILES
- Test parallel execution (both backends)

### Test Data
- Small protein PDB (~100 residues) for fast tests
- Set of ~10 known drug-like molecules
- Set of ~5 invalid/problematic SMILES

---

## Future Extensions

1. **Additional docking engines**: Gnina, smina (plugin architecture)
2. **GPU acceleration**: Gnina for CNN-based scoring
3. **Caching**: Hash-based caching of docking results
4. **Async API**: For integration with async RL frameworks
5. **Distributed Dask**: Multi-node cluster support