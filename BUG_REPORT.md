# DeepSC Bug Report

## Critical Bugs

### 1. Bare Exception Handler
**Location**: `src/deepsc/models/scbert/model.py` lines 16-19
```python
try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False
```
**Issue**: Bare `except:` catches all exceptions including system exits
**Fix**: Change to `except ImportError:`

### 2. Missing Error Handling in File Operations
**Location**: `src/deepsc/utils/utils.py` - `path_of_file()` function
**Issue**: Prints errors instead of raising exceptions, making error handling difficult
**Example**: Lines 49, 68, 70 use `print()` for errors
**Fix**: Raise appropriate exceptions (FileNotFoundError, ValueError)

### 3. Incorrect Test File
**Location**: `tests/test_expr_emb.py`
**Issue**: Contains unrelated code (L0Gate, GumbellThreeWayClassifier classes) instead of actual tests
**Fix**: Move classes to appropriate module, write actual tests

### 4. Potential Device Mismatch
**Location**: `src/deepsc/data/dataset.py` line 25
```python
row = torch.cat((row, torch.tensor([0], dtype=torch.long, device=row.device)))
```
**Issue**: Creating tensor without ensuring device compatibility
**Fix**: Ensure consistent device handling

## High Priority Issues

### 1. Hardcoded Paths
**Locations**: 
- `src/deepsc/data/preprocessing/config.py`
- `src/deepsc/data/download/tripleca/config.py`
**Examples**:
```python
BASE_URL = "/home/angli/DeepSC/"
DATA_PATH_CELLXGENE = "/home/data/cxg/"
```
**Issue**: Will fail on different systems
**Fix**: Use environment variables or configuration files

### 2. Missing Input Validation
**Locations**: Multiple functions throughout codebase
**Examples**:
- `normalize_tensor()` - no validation of input shape/type
- `discretize_expression()` - no handling of edge cases
**Fix**: Add input validation and appropriate error messages

### 3. Inconsistent Dependency Versions
**Issue**: `requirements.txt` and `pyproject.toml` specify different versions
**Example**: 
- requirements.txt: `numpy==2.2.2`
- pyproject.toml: `numpy==2.2.3`
**Fix**: Reconcile and use single source of truth

### 4. TODO Comments Indicating Incomplete Code
**Count**: 11 TODO comments found
**Notable Examples**:
- `src/deepsc/data/dataset.py` lines 17-24: Multiple TODOs about implementation decisions
- Performance optimization TODOs not addressed
**Fix**: Address or create issues for each TODO

## Medium Priority Issues

### 1. Wildcard Imports
**Locations**:
- `src/deepsc/models/scbert/__init__.py`
- `src/deepsc/train/trainer.py`
- `src/deepsc/utils/__init__.py`
**Issue**: Makes it unclear what's being imported, can cause namespace pollution
**Fix**: Use explicit imports

### 2. Print Statements Instead of Logging
**Count**: 17 files using `print()`
**Issue**: No control over output, not suitable for production
**Fix**: Replace with proper logging

### 3. Empty Files
**Locations**:
- `src/deepsc/__init__.py`
- `src/deepsc/evaluation/evaluator.py`
**Issue**: Unclear if intentional or incomplete
**Fix**: Add appropriate content or remove

### 4. Duplicate Functions
**Location**: `src/deepsc/utils/utils.py`
**Examples**:
- `setup_logging()` vs `set_log()`
- `save_ckpt()` vs `save_ckpt_fabric()`
**Fix**: Consolidate into single functions with parameters

## Low Priority Issues

### 1. Missing Type Hints
**Scope**: Most functions lack type annotations
**Impact**: Harder to understand function contracts
**Fix**: Add type hints progressively

### 2. Inconsistent Naming Conventions
**Examples**:
- `logfileName` (camelCase) vs `log_file_folder` (snake_case) in same function
- Inconsistent abbreviations: `cxg` vs `cellxgene`
**Fix**: Standardize to snake_case for functions/variables

### 3. No Test Coverage
**Issue**: Only one test file with no actual tests
**Impact**: Cannot verify refactoring doesn't break functionality
**Fix**: Add comprehensive test suite

### 4. Missing Docstrings
**Scope**: Many functions and classes lack documentation
**Impact**: Harder to understand and use the code
**Fix**: Add docstrings following NumPy style

## Potential Performance Issues

### 1. Inefficient Data Processing
**Location**: `src/deepsc/data/dataset.py` - `extract_rows_from_sparse_tensor_slow()`
**Issue**: Function name indicates known performance problem
**Fix**: Optimize or use the faster alternative already implemented

### 2. Repeated Computations
**Location**: `src/deepsc/data/dataset.py` line 23
**Comment**: "TODO: 和yongli确认，如果在这里激normalization的话比较低效，会造成重复计算"
**Fix**: Move normalization to preprocessing step

## Security Concerns

### 1. No Input Sanitization
**Locations**: Download functions, file path operations
**Risk**: Path traversal, command injection
**Fix**: Add input validation and sanitization

### 2. Direct String Formatting in Logs
**Risk**: Log injection if user input is logged
**Fix**: Use parameterized logging

## Recommendations

1. **Immediate Actions**:
   - Fix bare exception handler
   - Replace hardcoded paths
   - Add basic error handling

2. **Before Refactoring**:
   - Create comprehensive test suite
   - Document current behavior
   - Set up CI/CD pipeline

3. **During Refactoring**:
   - Fix bugs as part of refactoring
   - Add tests for each fix
   - Update documentation

---

*Note: This bug report should be updated as new issues are discovered or existing ones are resolved.*