#!/usr/bin/env bash
set -e

# If GOOGLE_CREDENTIALS_JSON env var exists, write it to file and set GOOGLE_APPLICATION_CREDENTIALS
if [ -n "${GOOGLE_CREDENTIALS_JSON:-}" ]; then
  echo "$GOOGLE_CREDENTIALS_JSON" > /app/google-credentials.json
  export GOOGLE_APPLICATION_CREDENTIALS="/app/google-credentials.json"
fi

# Optional: write other env files if needed (example for .env)
if [ -n "${DOTENV_CONTENT:-}" ]; then
  echo "$DOTENV_CONTENT" > /app/.env
fi

# Default CHATBOT_URL fallback (override with env var in Render)
: "${CHATBOT_URL:=http://localhost/CompanyWeb1/chatbot-api.php}"

# Use PORT env var if provided (Render sets PORT automatically)
: "${PORT:=8000}"

exec uvicorn main:app --host 0.0.0.0 --port "$PORT"
