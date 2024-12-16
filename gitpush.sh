#!/bin/bash

# Add all changes
git add .

# Get the commit message from the user
if [ "$1" ]; then
    # If message was passed as argument, use it
    message="$1"
else
    # Otherwise, prompt for a message
    echo "Enter commit message:"
    read message
fi

# Commit with the message
git commit -m "$message"

# Push to remote
git push origin main

echo "✅ Changes committed and pushed successfully!"
