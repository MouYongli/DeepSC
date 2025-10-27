# DeepSC Test Suite

This directory contains the test suite for DeepSC, organized by test type:

## Structure

- `unit/` - Unit tests for individual functions and classes
- `integration/` - Integration tests for end-to-end workflows  
- `conftest.py` - Pytest configuration and shared fixtures

## Running Tests

Run all tests:
```bash
pytest tests/
```

Run only unit tests:
```bash
pytest tests/unit/
```

Run only integration tests:
```bash
pytest tests/integration/
```

Run with coverage:
```bash
pytest tests/ --cov=src/deepsc
```

## Test Categories

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test complete workflows like data download and pretraining pipelines