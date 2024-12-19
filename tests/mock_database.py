class MockDatabaseManager:
    """Mock database manager for testing"""
    def __init__(self):
        self.data = {}
    
    def get_startup_data(self):
        """Return mock startup data"""
        return {
            'industry': 'healthcare technology',
            'stage': 'seed',
            'description': 'AI-powered healthcare platform'
        }
    
    def save_vc_data(self, vc_data):
        """Mock saving VC data"""
        self.data['vc_data'] = vc_data
        return True
    
    def get_vc_data(self):
        """Mock getting VC data"""
        return self.data.get('vc_data', [])
