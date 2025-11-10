# Contributing to M²FMoE

Thank you for your interest in contributing to M²FMoE! This document provides guidelines and instructions for contributing.

## Code of Conduct

We are committed to providing a welcoming and inclusive experience for everyone. Please be respectful and considerate in all interactions.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the bug
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment details (OS, Python version, PyTorch version)

### Suggesting Enhancements

We welcome suggestions for new features or improvements:
- Describe the enhancement clearly
- Explain why it would be useful
- Provide examples if possible

### Pull Requests

1. **Fork the repository**
   ```bash
   git clone https://github.com/Yaohui-Huang/M2FMoE.git
   cd M2FMoE
   ```

2. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Write clear, documented code
   - Follow existing code style
   - Add tests for new features
   - Update documentation

4. **Test your changes**
   ```bash
   python tests/test_model.py
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**
   - Describe your changes clearly
   - Reference any related issues
   - Wait for review

## Development Setup

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- Git

### Installation

```bash
# Clone repository
git clone https://github.com/Yaohui-Huang/M2FMoE.git
cd M2FMoE

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -e .[dev]
```

## Code Style

### Python Style Guide

We follow PEP 8 with some modifications:
- Line length: 100 characters (soft limit)
- Use 4 spaces for indentation
- Use docstrings for all public functions and classes

### Example Code Style

```python
def example_function(param1: int, param2: str) -> bool:
    """
    Brief description of what the function does.
    
    Args:
        param1 (int): Description of param1
        param2 (str): Description of param2
        
    Returns:
        bool: Description of return value
    """
    # Implementation
    result = some_operation(param1, param2)
    return result
```

## Project Structure

```
M2FMoE/
├── src/m2fmoe/          # Main source code
│   ├── models/          # Model implementations
│   ├── layers/          # Layer implementations
│   └── utils/           # Utility functions
├── scripts/             # Training and evaluation scripts
├── tests/               # Unit tests
├── configs/             # Configuration files
└── docs/                # Documentation
```

## Testing Guidelines

### Writing Tests

- Place tests in `tests/` directory
- Use descriptive test names
- Test edge cases
- Ensure tests are reproducible

### Running Tests

```bash
# Run all tests
python tests/test_model.py

# Run specific test
python -m pytest tests/test_model.py::test_m2fmoe_model
```

## Documentation Guidelines

### Docstring Format

Use Google-style docstrings:

```python
def function(arg1: type, arg2: type) -> return_type:
    """
    One-line summary.
    
    Longer description if needed.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When something is wrong
        
    Example:
        >>> function(1, "test")
        result
    """
```

### README Updates

When adding features:
- Update README.md with new functionality
- Add examples if applicable
- Update feature list

## Areas for Contribution

### High Priority

- [ ] Add more benchmark datasets
- [ ] Implement distributed training
- [ ] Add model compression techniques
- [ ] Improve documentation
- [ ] Add more visualization tools

### Medium Priority

- [ ] Add more evaluation metrics
- [ ] Implement online learning
- [ ] Add uncertainty quantification
- [ ] Create Jupyter notebook tutorials
- [ ] Add model interpretability tools

### Low Priority

- [ ] Add support for other frameworks (TensorFlow, JAX)
- [ ] Create web interface
- [ ] Add automatic hyperparameter tuning
- [ ] Implement model ensembling

## Commit Message Guidelines

Use clear, descriptive commit messages:

```
Type: Brief description (50 chars or less)

Detailed explanation if needed (72 chars per line).

- Bullet points for multiple changes
- Reference issues: Fixes #123

Types: Add, Update, Fix, Remove, Refactor, Docs, Test
```

### Examples

```
Add: Multi-scale attention mechanism

Implement multi-scale attention to capture patterns at different
temporal resolutions.

- Add MultiScaleAttention layer
- Update model architecture
- Add corresponding tests

Fixes #42
```

## Review Process

### What We Look For

- **Correctness**: Does the code work as intended?
- **Quality**: Is the code well-written and maintainable?
- **Tests**: Are there adequate tests?
- **Documentation**: Is the code properly documented?
- **Style**: Does it follow project conventions?

### Review Timeline

- Initial review: Within 1 week
- Follow-up: Within 3 days
- Merge: After approval from maintainer

## Questions?

If you have questions:
- Check existing issues and documentation
- Create a new issue with the "question" label
- Reach out to maintainers

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Acknowledgments

Thank you for contributing to M²FMoE! Your contributions help make this project better for everyone.

---

**Happy Contributing!** 🎉
