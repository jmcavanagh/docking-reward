# Contributing

Contributions are welcome! Here's how to get started.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/docking-reward.git
cd docking-reward

# Create conda environment with dependencies
conda create -n docking-reward-dev python=3.11
conda activate docking-reward-dev
conda install -c conda-forge vina openbabel plip rdkit meeko numpy pyyaml pytest

# Install in development mode
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Add docstrings for public functions and classes

## Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run tests to ensure everything works
5. Commit your changes (`git commit -am 'Add my feature'`)
6. Push to your fork (`git push origin feature/my-feature`)
7. Open a Pull Request

## Reporting Issues

Please include:
- Python version and OS
- Conda environment details (`conda list`)
- Full error traceback
- Minimal example to reproduce the issue
