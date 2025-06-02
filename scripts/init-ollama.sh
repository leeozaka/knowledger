#!/bin/bash

# Wait for Ollama to be ready
echo "Waiting for Ollama to be ready..."
until ollama list > /dev/null 2>&1; do
  sleep 2
  echo "Still waiting for Ollama..."
done

echo "Ollama is ready! Pulling gemma3:1b model..."

ollama pull gemma3:1b

echo "Model gemma3:1b has been pulled successfully!"

echo "Available models:"
ollama list 