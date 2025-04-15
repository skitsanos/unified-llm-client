# Publishing to PyPI

This document outlines the steps to publish the `unified-llm-client` package to PyPI.

## Prerequisites

1. PyPI account with 2FA enabled
2. Python 3.8 or higher
3. Required tools: `build` and `twine`

## Build and Publish Process

### 1. Install required tools

```bash
pip install build twine
```

### 2. Clean previous builds

```bash
rm -rf dist/ build/ *.egg-info/
```

### 3. Build the package

```bash
python -m build --sdist --wheel .
```

### 4. Check the package

```bash
python -m twine check dist/*
```

### 5. Test the package locally (optional)

```bash
pip install dist/unified_llm_client-X.X.X-py3-none-any.whl
```

Replace X.X.X with the actual version.

### 6. Upload to PyPI with 2FA

Since you have 2FA enabled on your PyPI account, you'll need to use an API token:

a. Go to your PyPI account settings: https://pypi.org/manage/account/
b. Under "API tokens", create a new token with appropriate scope (usually "Entire account")
c. Copy the token

Then use the token for authentication:

```bash
python -m twine upload dist/* -u __token__ -p <your-token>
```

Replace `<your-token>` with the actual token value.

### 7. Alternative: Use the Build Script

You can also use the provided build script:

```bash
python build_package.py
```

Then upload manually:

```bash
python -m twine upload dist/* -u __token__ -p <your-token>
```

## Testing on TestPyPI (Optional)

If you want to test the package before publishing to the main PyPI:

```bash
python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/* -u __token__ -p <your-test-token>
```

You'll need a separate token for TestPyPI.

## Version Management

Remember to update the version number in `pyproject.toml` before building a new release.

## Troubleshooting

- If you encounter authentication issues, ensure you're using the correct token
- If the upload fails, check for any validation errors in the twine output
- Make sure you're not trying to upload a version that already exists on PyPI
