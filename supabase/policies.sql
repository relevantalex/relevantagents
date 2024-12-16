-- Enable RLS
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE startups ENABLE ROW LEVEL SECURITY;
ALTER TABLE analyses ENABLE ROW LEVEL SECURITY;

-- Drop existing policies
DROP POLICY IF EXISTS "Enable all operations" ON documents;
DROP POLICY IF EXISTS "Enable all operations" ON startups;
DROP POLICY IF EXISTS "Enable all operations" ON analyses;

-- Create simple policies that enable all operations for authenticated users
CREATE POLICY "Enable all operations"
ON startups FOR ALL
TO authenticated
USING (true)
WITH CHECK (true);

CREATE POLICY "Enable all operations"
ON documents FOR ALL
TO authenticated
USING (true)
WITH CHECK (true);

CREATE POLICY "Enable all operations"
ON analyses FOR ALL
TO authenticated
USING (true)
WITH CHECK (true);

-- Grant necessary permissions
GRANT ALL ON documents TO authenticated;
GRANT ALL ON startups TO authenticated;
GRANT ALL ON analyses TO authenticated;
