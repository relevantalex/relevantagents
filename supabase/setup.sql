-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Drop existing tables if they exist
DROP TABLE IF EXISTS documents CASCADE;
DROP TABLE IF EXISTS analyses CASCADE;
DROP TABLE IF EXISTS startups CASCADE;

-- Create tables
CREATE TABLE startups (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    pitch TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    startup_id UUID REFERENCES startups(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    content TEXT,
    file_path TEXT,
    type TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE analyses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    startup_id UUID REFERENCES startups(id) ON DELETE CASCADE,
    analysis_type TEXT NOT NULL,
    data JSONB,
    status TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create trigger function for updating updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers
CREATE TRIGGER update_startups_updated_at
    BEFORE UPDATE ON startups
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_analyses_updated_at
    BEFORE UPDATE ON analyses
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Enable RLS
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE startups ENABLE ROW LEVEL SECURITY;
ALTER TABLE analyses ENABLE ROW LEVEL SECURITY;

-- Policy for documents table
CREATE POLICY "Enable insert for authenticated users only"
ON documents FOR INSERT
TO authenticated
WITH CHECK (true);

CREATE POLICY "Enable read access for all users"
ON documents FOR SELECT
TO authenticated
USING (true);

-- Policy for startups table
CREATE POLICY "Enable insert for authenticated users only"
ON startups FOR INSERT
TO authenticated
WITH CHECK (true);

CREATE POLICY "Enable read access for all users"
ON startups FOR SELECT
TO authenticated
USING (true);

-- Policy for analyses table
CREATE POLICY "Enable insert for authenticated users only"
ON analyses FOR INSERT
TO authenticated
WITH CHECK (true);

CREATE POLICY "Enable read access for all users"
ON analyses FOR SELECT
TO authenticated
USING (true);

-- Grant necessary permissions
GRANT ALL ON documents TO authenticated;
GRANT ALL ON startups TO authenticated;
GRANT ALL ON analyses TO authenticated;
