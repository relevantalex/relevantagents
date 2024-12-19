# Pre-Push Checklist

## Security
- [ ] No API keys in code (check all `.py` files)
- [ ] No hardcoded credentials or secrets
- [ ] `.env` and `.streamlit/secrets.toml` are in `.gitignore`

## Code Cleanup
- [ ] Remove debug print statements
- [ ] Clear test outputs in notebooks (if any)
- [ ] Remove commented-out code blocks (unless needed for future reference)

## Testing
- [ ] Run test suite: `python tests/test_vc_scraper.py`
- [ ] Test with mock API keys to ensure key handling works

## Files to Keep
- Keep test files and mocks (useful for future testing)
- Keep documentation files
- Keep example configurations (with placeholder values)

## Quick Commands
```bash
# Check for sensitive data
grep -r "api_key\|secret\|password" .

# Run tests
python tests/test_vc_scraper.py

# Check git status
git status
```
