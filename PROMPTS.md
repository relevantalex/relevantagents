# AI Agent Prompts and Best Practices

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
