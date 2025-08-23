# Contributing to DeepSC

Thank you for your interest in contributing to DeepSC! This guide will help you get started.

## Getting Started

### Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MouYongli/DeepSC.git
   cd DeepSC
   ```

2. **Create a conda environment:**
   ```bash
   conda create --name deepsc-dev python=3.10
   conda activate deepsc-dev
   ```

3. **Install dependencies:**
   ```bash
   # Install PyTorch with CUDA support
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   pip install torch_geometric
   
   # Install the package in development mode
   pip install -e .
   
   # Install development dependencies
   pip install pytest pytest-cov black isort flake8
   ```

4. **Set up pre-commit hooks:**
   ```bash
   pre-commit install
   ```

### Project Structure

Understanding the codebase organization:

```
src/deepsc/
├── data/              # Data loading and preprocessing
│   ├── download/      # Data download scripts 
│   └── preprocessing/ # Data preprocessing utilities
├── models/            # Model architectures
├── train/             # Training infrastructure
├── pretrain/          # Pretraining entry points
├── finetune/          # Finetuning entry points
└── utils/             # Utility functions

configs/               # Hydra configuration files
tests/                 # Test suite
scripts/debug/         # Development and debugging utilities
```

## Development Workflow

### 1. Before Making Changes

- Run existing tests to ensure everything works:
  ```bash
  pytest tests/ -v
  ```

- Run integration tests to verify pipelines:
  ```bash
  pytest tests/integration/ -v
  ```

### 2. Making Changes

- **Follow naming conventions:**
  - Use `snake_case` for functions and variables
  - Use `PascalCase` for classes
  - Use lowercase with hyphens for directories

- **Write tests for new functionality:**
  - Unit tests go in `tests/unit/`
  - Integration tests go in `tests/integration/`

- **Update documentation:**
  - Update docstrings for new functions
  - Update README.md if adding new features
  - Add configuration examples for new models/datasets

### 3. Code Quality

- **Format your code:**
  ```bash
  black src/ tests/
  isort src/ tests/
  ```

- **Run linters:**
  ```bash
  flake8 src/ tests/
  ```

- **Run tests:**
  ```bash
  pytest tests/ -v --cov=src/deepsc
  ```

## Adding New Components

### Adding a New Model

1. Create model implementation in `src/deepsc/models/your_model/`
2. Add configuration in `configs/pretrain/model/your_model.yaml`
3. Add tests in `tests/unit/test_your_model.py`
4. Update documentation

### Adding a New Dataset

1. Create dataset class in `src/deepsc/data/`
2. Add preprocessing scripts if needed
3. Add configuration in `configs/pretrain/dataset/your_dataset.yaml`
4. Add tests and documentation

### Adding Data Download Scripts

1. Create download script in `src/deepsc/data/download/your_source/`
2. Add configuration in the script's config.py
3. Add documentation in a README.md file
4. Test the download pipeline

## Testing Guidelines

### Unit Tests
- Test individual functions in isolation
- Mock external dependencies
- Aim for good test coverage

### Integration Tests  
- Test complete workflows
- Test data pipeline from download to preprocessing
- Test model training/inference pipelines

### Running Tests
```bash
# All tests
pytest tests/

# Unit tests only
pytest tests/unit/

# Integration tests only  
pytest tests/integration/

# With coverage report
pytest tests/ --cov=src/deepsc --cov-report=html
```

## Pull Request Guidelines

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes and commit:**
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```

3. **Push and create a pull request:**
   ```bash
   git push origin feature/your-feature-name
   ```

4. **PR Requirements:**
   - All tests must pass
   - Code must be formatted with black and isort
   - Add tests for new functionality
   - Update documentation as needed
   - Write clear commit messages

## Configuration Management

We use Hydra for configuration management. When adding new configurations:

1. **Follow the existing structure:**
   - Model configs in `configs/pretrain/model/`
   - Dataset configs in `configs/pretrain/dataset/`
   - Main configs in `configs/pretrain/`

2. **Use `_target_` for object instantiation:**
   ```yaml
   _target_: deepsc.models.your_model.YourModel
   param1: value1
   param2: value2
   ```

3. **Document configuration options:**
   - Add comments explaining parameters
   - Provide example values
   - Document valid parameter ranges

## Getting Help

- **Issues:** Create a GitHub issue for bugs or feature requests
- **Discussions:** Use GitHub Discussions for questions
- **Documentation:** Check existing documentation and code examples

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and contribute
- Follow best practices for scientific software development

Thank you for contributing to DeepSC!