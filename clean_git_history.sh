#!/bin/bash

# Use BFG Repo Cleaner to remove the API key
bfg --replace-text keys.txt

# Force push the changes
git reflog expire --expire=now --all && git gc --prune=now --aggressive
git push --force
