# Contributing to CITEDiscord-Net

We welcome contributions to CITEDiscord-Net! This document provides guidelines for contributing.

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/CITEDiscord-Net.git
   cd CITEDiscord-Net
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Code Style

- Follow PEP 8 conventions
- Use type hints where possible
- Add docstrings to all public functions and classes
- Keep functions focused and modular

## Submitting Changes

1. Commit your changes with clear, descriptive messages
2. Push your branch to your fork
3. Open a Pull Request against `main`
4. Describe your changes and link any relevant issues

## Reporting Issues

- Use GitHub Issues to report bugs or request features
- Include reproduction steps, expected vs actual behaviour, and system info

## Areas for Contribution

- Support for additional CITE-seq datasets
- Larger antibody panels (ASAP-seq, TEA-seq)
- Spatial CITE-seq integration
- Supervised GAT training for communication refinement
- Foundation model pretraining (scGPT, Geneformer integration)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
