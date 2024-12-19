# AI Agent Prompts and Best Practices

⚠️ **CRITICAL SECURITY WARNING** ⚠️
## NEVER EVER COMMIT API KEYS OR SECRETS
- NEVER put API keys in code
- NEVER commit .env files
- NEVER share keys in documentation
- NEVER use real keys in examples
- ALWAYS check files before committing
- ALWAYS use environment variables or secrets management
- ALWAYS rotate keys if accidentally exposed

## General Guidelines

### 1. Testing and Verification
- Always test new features in a terminal/script before implementing in Streamlit
- Verify API keys and credentials work before full implementation
- Use proper error handling and logging from the start
- Add progress indicators for long-running operations

### 2. API Integration Best Practices
```python
# SSL Context Setup
ssl_context = ssl.create_default_context(cafile=certifi.where())
conn = aiohttp.TCPConnector(ssl=ssl_context)

# Session Management
async with aiohttp.ClientSession(connector=conn) as session:
    async with session.get(url, params=params) as response:
        if response.status == 200:
            data = await response.json()
```

### 3. Error Handling Template
```python
try:
    # Main operation
    result = await some_operation()
    if result:
        st.write(f"✅ Success: {result}")
    else:
        st.warning("⚠️ No results found")
except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    logger.error(f"Operation failed: {str(e)}")
```

### 4. Progress Tracking
```python
with st.spinner("🔄 Processing..."):
    progress_bar = st.progress(0)
    for idx, item in enumerate(items):
        # Process item
        progress_bar.progress((idx + 1) / len(items))
```

## Search Agent Patterns

### 1. Search Templates
```python
search_templates = {
    'linkedin_company': 'site:linkedin.com/company "{industry}" "venture capital"',
    'linkedin_people': 'site:linkedin.com/in "{industry}" "venture capital" "partner"',
    'crunchbase': 'site:crunchbase.com/organization "{industry}" "venture capital"'
}
```

### 2. Rate Limiting
```python
# Rate limit handling
if response.status == 429:
    st.warning("⚠️ Rate limit reached. Waiting...")
    await asyncio.sleep(2)
    continue

# General rate limiting
await asyncio.sleep(0.5)  # Respect API limits
```

## GPT Integration Patterns

### 1. Validation Prompts
```python
prompt = f"""
Analyze this data and extract information:
Input: {data}
Required fields:
1. Field one
2. Field two
3. Relevance score (0-1)

Return JSON format only.
"""
```

### 2. Response Parsing
```python
try:
    validation_data = json.loads(response.choices[0].message.content)
    if validation_data.get('relevance_score', 0) > threshold:
        return process_validation(validation_data)
except json.JSONDecodeError:
    st.error("Failed to parse GPT response")
```

## Common UI Patterns

### 1. Status Messages
- ✅ Success: Green success messages
- ⚠️ Warning: Yellow warning messages
- ❌ Error: Red error messages
- 🔄 Processing: Blue info messages

### 2. Progress Indicators
- Use spinners for unknown duration tasks
- Use progress bars for tasks with known steps
- Show step counts (e.g., "Step 2/5")

## Security Best Practices

### 1. API Key Management
- Always use Streamlit secrets for API keys
- Never hardcode sensitive information
- Validate API keys before using them

### 2. Error Messages
- Show user-friendly error messages
- Log detailed errors for debugging
- Don't expose sensitive information in error messages

## Critical Security Rules

### API Key Management - READ THIS FIRST
1. **🚫 ABSOLUTELY NO API KEYS IN CODE**
   ```python
   # ❌ NEVER DO THIS - SECURITY RISK!
   api_key = "your_key_here"
   
   # ✅ ALWAYS DO THIS
   api_key = st.secrets["api_keys"]["key_name"]
   # OR
   api_key = os.getenv("API_KEY")
   ```

2. **Pre-Commit Security Checks**
   - Run `git diff` before every commit
   - Use automated pre-commit hooks
   - Review all files for secrets
   - When in doubt, DON'T commit

3. **If Keys Are Exposed:**
   1. IMMEDIATELY revoke/delete the exposed keys
   2. Generate new keys with restrictions
   3. Update all services using the old keys
   4. Clean git history
   5. Review security practices

4. **Secure Testing:**
   ```python
   # ❌ NEVER
   TEST_API_KEY = "actual-api-key"
   
   # ✅ ALWAYS
   TEST_API_KEY = os.getenv("TEST_API_KEY", "dummy-key")
   ```

### Security Checklist
Before EVERY commit:
- [ ] No API keys in code
- [ ] No secrets in documentation
- [ ] No .env files
- [ ] No configuration files with real credentials
- [ ] No test files with real keys

## Testing Checklist

1. **Before Implementation**
   - [ ] Test API connectivity
   - [ ] Verify credentials
   - [ ] Check rate limits
   - [ ] Test sample queries

2. **During Implementation**
   - [ ] Add proper error handling
   - [ ] Implement progress tracking
   - [ ] Add user feedback
   - [ ] Test edge cases

3. **After Implementation**
   - [ ] Verify all features work
   - [ ] Check error handling
   - [ ] Test with various inputs
   - [ ] Verify rate limiting works

## Pre-commit Checks
```bash
# Add to .git/hooks/pre-commit
#!/bin/bash

# Check for API keys
if grep -r "api[_-]key.*=.*[A-Za-z0-9_-]\{20,\}" .; then
    echo "Potential API key found. Commit rejected."
    exit 1
fi
