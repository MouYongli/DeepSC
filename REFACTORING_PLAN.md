# DeepSC Repository Refactoring Plan

## Executive Summary

This document outlines a comprehensive refactoring plan for the DeepSC repository to improve code quality, maintainability, and reliability. The plan focuses on addressing identified bugs, improving code structure, and ensuring all functionality remains intact.

## Current State Analysis

### Repository Structure
- **Language**: Python 3.10+
- **Framework**: PyTorch 2.6.0 with CUDA 12.6
- **Configuration**: Hydra-based configuration management
- **Key Components**: Data processing, model training, evaluation pipelines

### Identified Issues

#### 1. Code Quality Issues
- **Duplicate Code**: Multiple functions with similar functionality (e.g., logging setup, checkpoint saving)
- **Inconsistent Naming**: Mixed camelCase and snake_case conventions
- **Dead Code**: Unused imports, empty files, test functions
- **TODO Comments**: 11 unresolved TODO comments indicating incomplete implementations

#### 2. Bugs and Potential Issues
- **Exception Handling**: Bare `except:` clause in `scbert/model.py` (lines 16-19)
- **Import Errors**: Wildcard imports in multiple files
- **Configuration Issues**: Hardcoded paths in config files
- **Missing Error Handling**: Critical functions without proper exception handling
- **Print Statements**: 17 files using print() instead of proper logging

#### 3. Testing and Documentation
- **No Tests**: Only one test file (`test_expr_emb.py`) with incomplete/commented code
- **Missing Documentation**: Many functions lack docstrings
- **No Type Hints**: Most functions missing type annotations

#### 4. Dependency Management
- **Version Conflicts**: `requirements.txt` and `pyproject.toml` have different dependency versions
- **Missing Dependencies**: Some imports not listed in requirements

## Refactoring Plan

### Phase 1: Critical Bug Fixes and Safety Improvements (Week 1)

#### 1.1 Fix Exception Handling
- [ ] Replace bare `except:` with specific exception handling in `scbert/model.py`
- [ ] Add proper error handling to critical functions:
  - `path_of_file()` in `utils/utils.py`
  - `normalize_tensor()` in `data/dataset.py`
  - File I/O operations in download scripts

#### 1.2 Remove Security Risks
- [ ] Add input validation for file paths
- [ ] Sanitize user inputs in download functions
- [ ] Replace hardcoded paths with environment variables

#### 1.3 Fix Import Issues
- [ ] Remove wildcard imports
- [ ] Clean up unused imports
- [ ] Organize imports according to PEP 8

### Phase 2: Code Structure Improvements (Week 2)

#### 2.1 Consolidate Duplicate Code
- [ ] Merge `setup_logging()` and `set_log()` into single function
- [ ] Consolidate `save_ckpt()` and `save_ckpt_fabric()` 
- [ ] Unify dataset classes where appropriate

#### 2.2 Standardize Naming Conventions
- [ ] Convert all function/variable names to snake_case
- [ ] Standardize abbreviations (e.g., use either `cxg` or `cellxgene` consistently)
- [ ] Update class names to follow PascalCase

#### 2.3 Remove Dead Code
- [ ] Delete `testPackage()` function
- [ ] Remove empty files or add appropriate content
- [ ] Clean up commented-out code

### Phase 3: Configuration and Dependency Management (Week 3)

#### 3.1 Centralize Configuration
- [ ] Create central configuration module
- [ ] Move all hardcoded values to config files
- [ ] Use environment variables for sensitive data
- [ ] Consolidate duplicate configuration values

#### 3.2 Fix Dependency Management
- [ ] Reconcile `requirements.txt` and `pyproject.toml`
- [ ] Pin all dependency versions
- [ ] Add missing dependencies
- [ ] Create separate dev-requirements.txt

### Phase 4: Testing Infrastructure (Week 4)

#### 4.1 Set Up Testing Framework
- [ ] Configure pytest
- [ ] Create test directory structure
- [ ] Add pytest configuration file
- [ ] Set up test fixtures

#### 4.2 Write Unit Tests
- [ ] Core utility functions (priority: file operations, data processing)
- [ ] Dataset classes
- [ ] Model components
- [ ] Data preprocessing functions

#### 4.3 Integration Tests
- [ ] End-to-end training pipeline
- [ ] Data download and preprocessing
- [ ] Model inference

### Phase 5: Documentation and Type Safety (Week 5)

#### 5.1 Add Type Hints
- [ ] Add type hints to all functions
- [ ] Use typing module for complex types
- [ ] Configure mypy for type checking

#### 5.2 Write Documentation
- [ ] Add docstrings to all functions/classes
- [ ] Update README with setup instructions
- [ ] Create API documentation
- [ ] Document configuration options

### Phase 6: Code Quality Tools (Week 6)

#### 6.1 Set Up Linting
- [ ] Configure flake8/ruff
- [ ] Set up black for formatting
- [ ] Configure isort for imports
- [ ] Add pre-commit hooks

#### 6.2 Continuous Integration
- [ ] Set up GitHub Actions
- [ ] Add test automation
- [ ] Add code coverage reporting
- [ ] Configure automatic linting

## Implementation Strategy

### Approach
1. **Branch Strategy**: Create feature branches for each phase
2. **Testing**: Test each change thoroughly before merging
3. **Review**: Code review for all changes
4. **Documentation**: Update docs with each change

### Priority Order
1. Critical bug fixes (Phase 1) - **HIGHEST PRIORITY**
2. Testing infrastructure (Phase 4) - **HIGH PRIORITY**
3. Code structure improvements (Phase 2) - **MEDIUM PRIORITY**
4. Configuration management (Phase 3) - **MEDIUM PRIORITY**
5. Documentation (Phase 5) - **LOW PRIORITY**
6. Code quality tools (Phase 6) - **LOW PRIORITY**

## Testing Strategy

### Unit Testing
- Test each function in isolation
- Mock external dependencies
- Aim for 80% code coverage

### Integration Testing
- Test complete workflows
- Verify data pipeline integrity
- Test model training/inference

### Regression Testing
- Create baseline results before refactoring
- Compare outputs after each change
- Ensure numerical stability

## Risk Mitigation

### Backup Strategy
- Create full backup before starting
- Tag current version in git
- Document current behavior

### Rollback Plan
- Keep old code in separate branch
- Test thoroughly before removing old code
- Maintain compatibility during transition

### Communication
- Document all changes
- Update team on progress
- Get approval for breaking changes

## Success Metrics

### Code Quality
- [ ] Zero linting errors
- [ ] 80%+ test coverage
- [ ] All functions have type hints
- [ ] All functions have docstrings

### Functionality
- [ ] All existing features work
- [ ] No performance regression
- [ ] Improved error messages
- [ ] Better logging

### Maintainability
- [ ] Clear code structure
- [ ] Consistent naming
- [ ] No duplicate code
- [ ] Comprehensive tests

## Timeline

- **Week 1**: Critical fixes and safety improvements
- **Week 1**: Code structure refactoring
- **Week 1**: Configuration management
- **Week 2**: Testing infrastructure
- **Week 2**: Documentation and types
- **Week 2**: Quality tools and CI/CD

Total estimated time: 2 weeks

## Next Steps

1. Review and approve this plan
2. Create feature branches
3. Start with Phase 1 critical fixes
4. Set up regular progress meetings
5. Begin implementation

---

*This plan is a living document and will be updated as the refactoring progresses.*