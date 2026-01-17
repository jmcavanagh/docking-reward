"""Configuration parsing and validation for docking reward calculator."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# Valid interaction types
VALID_INTERACTION_TYPES = frozenset({
    "hydrogen_bond",
    "hydrophobic",
    "salt_bridge",
    "pi_stacking",
    "any",
})


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

    residue: str  # e.g., "ASP123"
    type: str  # hydrogen_bond, hydrophobic, salt_bridge, pi_stacking, any
    weight: float  # Score contribution per interaction count


@dataclass
class TargetConfig:
    """Configuration for a single docking target."""

    name: str
    pdb_path: Path
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    weight: float
    interactions: list[InteractionConfig] = field(default_factory=list)
    # Energy threshold for considering poses in interaction scoring (kcal/mol)
    # All poses within this threshold of the best pose are evaluated
    interaction_energy_threshold: float = 1.0
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
    custom: list[CustomScorerConfig] = field(default_factory=list)


@dataclass
class Config:
    """Root configuration object."""

    vina: VinaConfig
    druglikeness: DruglikenessConfig
    targets: list[TargetConfig]


def _parse_vina_config(data: dict) -> VinaConfig:
    """Parse Vina configuration section."""
    if not data:
        return VinaConfig()

    return VinaConfig(
        exhaustiveness=data.get("exhaustiveness", 8),
        n_poses=data.get("n_poses", 9),
        energy_range=data.get("energy_range", 3.0),
        seed=data.get("seed"),
    )


def _parse_interaction_config(data: dict) -> InteractionConfig:
    """Parse a single interaction configuration."""
    if "residue" not in data:
        raise ValueError("Interaction config missing required field 'residue'")
    if "type" not in data:
        raise ValueError("Interaction config missing required field 'type'")
    if "weight" not in data:
        raise ValueError("Interaction config missing required field 'weight'")

    interaction_type = data["type"]
    if interaction_type not in VALID_INTERACTION_TYPES:
        raise ValueError(
            f"Invalid interaction type '{interaction_type}'. "
            f"Must be one of: {', '.join(sorted(VALID_INTERACTION_TYPES))}"
        )

    return InteractionConfig(
        residue=data["residue"],
        type=interaction_type,
        weight=float(data["weight"]),
    )


def _parse_target_config(data: dict) -> TargetConfig:
    """Parse a single target configuration."""
    required_fields = ["name", "pdb_path", "center", "size", "weight"]
    for field_name in required_fields:
        if field_name not in data:
            raise ValueError(f"Target config missing required field '{field_name}'")

    center = data["center"]
    if not isinstance(center, list) or len(center) != 3:
        raise ValueError("Target 'center' must be a list of 3 floats [x, y, z]")

    size = data["size"]
    if not isinstance(size, list) or len(size) != 3:
        raise ValueError("Target 'size' must be a list of 3 floats [x, y, z]")

    interactions = [
        _parse_interaction_config(i) for i in data.get("interactions", [])
    ]

    return TargetConfig(
        name=data["name"],
        pdb_path=Path(data["pdb_path"]),
        center=tuple(float(x) for x in center),
        size=tuple(float(x) for x in size),
        weight=float(data["weight"]),
        interactions=interactions,
        interaction_energy_threshold=float(data.get("interaction_energy_threshold", 1.0)),
        exhaustiveness=data.get("exhaustiveness"),
        n_poses=data.get("n_poses"),
        energy_range=data.get("energy_range"),
    )


def _parse_druglikeness_config(data: dict) -> DruglikenessConfig:
    """Parse drug-likeness configuration section."""
    if not data:
        return DruglikenessConfig()

    qed_config = None
    if "qed" in data:
        qed_data = data["qed"]
        qed_config = QEDConfig(weight=float(qed_data.get("weight", 1.0)))

    custom_scorers = []
    for custom_data in data.get("custom", []):
        if "path" not in custom_data:
            raise ValueError("Custom scorer missing required field 'path'")
        if "function" not in custom_data:
            raise ValueError("Custom scorer missing required field 'function'")
        if "weight" not in custom_data:
            raise ValueError("Custom scorer missing required field 'weight'")

        custom_scorers.append(
            CustomScorerConfig(
                path=Path(custom_data["path"]),
                function=custom_data["function"],
                weight=float(custom_data["weight"]),
            )
        )

    return DruglikenessConfig(qed=qed_config, custom=custom_scorers)


def load_config(path: Path | str) -> Config:
    """
    Load and validate configuration from YAML file.

    Args:
        path: Path to YAML configuration file

    Returns:
        Validated Config object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid (missing required fields, invalid values)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if not data:
        raise ValueError("Configuration file is empty")

    # Parse sections
    vina_config = _parse_vina_config(data.get("vina", {}))
    druglikeness_config = _parse_druglikeness_config(data.get("druglikeness", {}))

    # Parse targets
    targets_data = data.get("targets", [])
    if not targets_data:
        raise ValueError("Configuration must define at least one target")

    targets = [_parse_target_config(t) for t in targets_data]

    config = Config(
        vina=vina_config,
        druglikeness=druglikeness_config,
        targets=targets,
    )

    # Validate semantic correctness
    validate_config(config)

    return config


def validate_config(config: Config) -> None:
    """
    Validate configuration for semantic correctness.

    Checks:
    - All PDB paths exist and are readable
    - Box dimensions are positive
    - Custom scorer paths exist
    - At least one target is defined

    Raises:
        ValueError: With descriptive message if validation fails
    """
    if not config.targets:
        raise ValueError("Configuration must define at least one target")

    for target in config.targets:
        # Check PDB path exists
        if not target.pdb_path.exists():
            raise ValueError(f"PDB file not found for target '{target.name}': {target.pdb_path}")

        # Check box dimensions are positive
        for i, dim in enumerate(["x", "y", "z"]):
            if target.size[i] <= 0:
                raise ValueError(
                    f"Target '{target.name}' has non-positive box size in {dim} dimension"
                )

    # Check custom scorer paths exist
    for custom in config.druglikeness.custom:
        if not custom.path.exists():
            raise ValueError(f"Custom scorer file not found: {custom.path}")
