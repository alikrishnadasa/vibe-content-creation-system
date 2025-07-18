#!/bin/bash

echo "🌐 Starting Video Content Creation Frontend"
echo "=========================================="

# Navigate to frontend directory
cd frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing npm dependencies..."
    npm install
fi

# Start the Next.js development server
echo "⚡ Starting Next.js development server on http://localhost:3000"
echo "🚀 Make sure the backend is running on http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

npm run dev