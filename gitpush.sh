#!/bin/bash

# Add all changes
git add .

# Get the commit message from the user
if [ "$1" ]; then
    # If message was passed as argument, use it
    message="$1"
else
    # Use default message with timestamp
    message="update $(date '+%Y-%m-%d %H:%M:%S')"
fi

# Commit with the message
git commit -m "$message"

# Push to remote
git push origin main

echo "✅ Changes committed and pushed successfully!"
