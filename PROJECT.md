# RelevantAgents Project Documentation

## Overview
RelevantAgents is a comprehensive business intelligence and lead generation platform that helps startups and businesses identify, analyze, and connect with relevant stakeholders in their industry. The platform uses advanced AI and web scraping techniques to gather and process information from various sources.

## Core Features

### 1. Online Competitor Analysis
- Automated competitor discovery using multiple search engines
- Detailed competitor analysis and profiling
- Market positioning insights
- Feature comparison matrix
- Pricing analysis

### 2. VC Discovery and Outreach
- Multi-agent system for VC firm discovery
- Comprehensive VC database integration (Crunchbase, Unicorn Nest)
- Automated relevance scoring
- Contact information extraction
- Email and LinkedIn profile discovery

### 3. Industry Analysis
- Market trend analysis
- Industry-specific insights
- Growth opportunity identification
- Competitive landscape mapping

## Technical Architecture

### Current Structure
```
relevantagents/
├── .streamlit/               # Streamlit configuration
├── pages/                    # Application pages
│   ├── Online_Competitor_List.py
│   ├── VC_Email_List_Scraper.py
│   ├── Industry_Analysis.py
│   └── Relevant_VC_Scraper.py
├── app.py                    # Main application file
├── database.py              # Database operations
├── document_processor.py    # Document processing utilities
└── requirements.txt         # Project dependencies
```

### Components

#### 1. Multi-Agent System
- **SearchAgent**: Handles initial discovery across multiple sources
- **EnrichmentAgent**: Processes and enriches gathered data
- **ContactAgent**: Extracts and validates contact information

#### 2. Data Sources
- DuckDuckGo Search
- Brave Search API
- Google Custom Search API
- Crunchbase API
- Unicorn Nest
- LinkedIn (planned)
- PitchBook (planned)

#### 3. Database
- Supabase integration
- Structured data storage
- Caching system for search results

## Development Timeline

### Phase 1: Foundation (Completed)
- ✅ Basic competitor analysis
- ✅ Initial VC scraping
- ✅ Streamlit UI implementation
- ✅ Database integration

### Phase 2: Enhanced VC Discovery (Current)
- ✅ Multi-agent system implementation
- ✅ Multiple search provider integration
- 🔄 API integrations
  - [ ] Brave Search
  - [ ] Google Custom Search
  - [ ] Crunchbase
- [ ] Improved contact discovery
- [ ] Enhanced relevance scoring

### Phase 3: Advanced Features (Q1 2024)
- [ ] Machine learning-based relevance scoring
- [ ] Automated email outreach
- [ ] Custom email templates
- [ ] Campaign tracking
- [ ] Integration with CRM systems

### Phase 4: Enterprise Features (Q2 2024)
- [ ] Team collaboration
- [ ] Advanced analytics
- [ ] Custom reporting
- [ ] API access
- [ ] White-label solutions

## API Keys and Configuration

### Current Configuration Structure
The application uses `.streamlit/secrets.toml` for configuration. Here's the current structure:

```toml
[api_keys]
brave_api_key = ""          # Brave Search API key
openai_api_key = ""         # OpenAI API key for GPT-4
anthropic_api_key = ""      # Anthropic API key for Claude
google_api_key = ""         # To be added: Google Custom Search
google_cx = ""             # To be added: Google Custom Search Engine ID
crunchbase_api_key = ""    # To be added: Crunchbase API

[api_settings]
ai_provider = "openai"     # Current AI provider (options: "openai", "anthropic")

[model_settings]
openai_model = "gpt-4"                        # OpenAI model selection
anthropic_model = "claude-3-opus-20240229"    # Anthropic model selection

[supabase]
url = ""                   # Supabase project URL
anon_key = ""             # Supabase anonymous key
service_email = ""        # Service account email
service_password = ""     # Service account password
service_role_key = ""     # Service role key for admin access
```

### Required API Keys
1. **Brave Search API**
   - Signup: https://brave.com/search/api/
   - Usage: Enhanced web search capabilities
   - Pricing: Free tier available

2. **Google Custom Search API**
   - Signup: https://developers.google.com/custom-search/v1/overview
   - Required: API key and Custom Search Engine ID (cx)
   - Pricing: 100 free queries/day

3. **Crunchbase API**
   - Signup: https://data.crunchbase.com/docs/using-the-api
   - Usage: VC firm data and investment information
   - Pricing: Enterprise pricing

### Configuration
API keys should be stored in `.streamlit/secrets.toml`:
```toml
[api_keys]
brave_api_key = "your_brave_api_key"
openai_api_key = "your_openai_api_key"
anthropic_api_key = "your_anthropic_api_key"
google_api_key = "your_google_api_key"
google_cx = "your_google_cx"
crunchbase_api_key = "your_crunchbase_api_key"
```

## Best Practices

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Document all functions and classes
- Handle exceptions gracefully
- Log errors and important events

### Data Processing
- Implement rate limiting for API calls
- Cache search results
- Validate and clean data before storage
- Handle duplicates efficiently
- Implement proper error handling

### Security
- Never commit API keys
- Validate user input
- Implement rate limiting
- Follow GDPR compliance
- Secure data storage

## Testing

### Unit Tests
- Test each agent independently
- Validate data processing functions
- Mock API responses
- Test error handling

### Integration Tests
- Test multi-agent interactions
- Validate data flow
- Test API integrations
- End-to-end testing

## Deployment

### Requirements
- Python 3.8+
- Streamlit
- Required Python packages in requirements.txt
- API keys configured in secrets.toml

### Setup Instructions
1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure API keys in `.streamlit/secrets.toml`
4. Run application: `streamlit run app.py`

## Contributing
1. Follow Git flow branching model
2. Write clear commit messages
3. Document changes in CHANGELOG.md
4. Submit pull requests with descriptions
5. Review code before merging

## Future Enhancements

### Short-term
- Implement remaining API integrations
- Enhance contact discovery
- Improve relevance scoring
- Add more data sources

### Long-term
- Machine learning models for scoring
- Automated outreach system
- Advanced analytics
- Enterprise features
- API access

## Maintenance

### Regular Tasks
- Update dependencies
- Monitor API usage
- Clean database
- Update documentation
- Review and optimize code

### Monitoring
- Track API usage
- Monitor error rates
- Check system performance
- Review user feedback

## Support

### Documentation
- Keep README updated
- Maintain API documentation
- Document common issues
- Provide setup guides

### User Support
- Monitor issues
- Provide timely responses
- Update FAQ
- Create tutorials

---

*Last Updated: 2024-12-19*
