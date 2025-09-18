#!/usr/bin/env bash
set -euo pipefail

# start_all.sh
# Helper script to start the Streamlit frontend and (optionally) open ngrok tunnels
# Usage:
#   ./start_all.sh
# Requirements:
#  - streamlit installed and available on PATH
#  - (optional) ngrok installed for public sharing
#  - (optional) a Python virtualenv at .venv (the script will activate it if present)

echo "=== Movie Recommender System helper ==="

# Activate virtualenv if present
if [ -f ".venv/bin/activate" ]; then
  echo "Activating .venv virtualenv"
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# Warn about Ollama dependency
if ! command -v ollama >/dev/null 2>&1; then
  echo "NOTE: 'ollama' is not found on PATH. If your app needs Ollama, start it separately."
else
  echo "ollama found on PATH. Ensure your Ollama server/model is running (default: http://localhost:11434)."
fi

# Start Streamlit
PORT=8501
ADDR=0.0.0.0
echo "Starting Streamlit on ${ADDR}:${PORT}"
streamlit run app.py --server.address ${ADDR} --server.port ${PORT} &
STREAMLIT_PID=$!
sleep 1

# Start ngrok tunnels if ngrok is installed
if command -v ngrok >/dev/null 2>&1; then
  echo "ngrok detected. Starting ngrok for Streamlit (port ${PORT})"
  # start ngrok in background and write logs to files
  ngrok http ${PORT} --log=stdout > ngrok_streamlit.log 2>&1 &
  NGROK_PID=$!
  sleep 2
  # attempt to extract public URL
  NGROK_URL=$(grep -Eo "https://[0-9a-zA-Z.-]+(\.ngrok.io|\.trycloudflare.com)?" ngrok_streamlit.log | head -n1 || true)
  if [ -n "${NGROK_URL}" ]; then
    echo "Streamlit public URL: ${NGROK_URL}"
  else
    echo "Unable to parse ngrok URL automatically. See ngrok_streamlit.log for details."
  fi
  # If OLLAMA is used locally, optionally open ngrok tunnel for Ollama too
  if command -v ollama >/dev/null 2>&1; then
    echo "Starting ngrok for Ollama (port 11434)"
    ngrok http 11434 --log=stdout > ngrok_ollama.log 2>&1 &
    NGROK_OLLAMA_PID=$!
    sleep 2
    NGROK_OLLAMA_URL=$(grep -Eo "https://[0-9a-zA-Z.-]+(\.ngrok.io|\.trycloudflare.com)?" ngrok_ollama.log | head -n1 || true)
    if [ -n "${NGROK_OLLAMA_URL}" ]; then
      echo "Ollama public URL: ${NGROK_OLLAMA_URL}"
      echo "If you expose Ollama, set the OLLAMA_URL environment variable before running the app, e.g."
      echo "  export OLLAMA_URL='${NGROK_OLLAMA_URL}'"
    else
      echo "Unable to parse ngrok Ollama URL automatically. See ngrok_ollama.log for details."
    fi
  fi
else
  echo "ngrok not found. To expose your app run 'ngrok http ${PORT}' after installing ngrok: https://ngrok.com/"
fi

echo "Started Streamlit (PID=${STREAMLIT_PID}). To stop everything run:"
echo "  kill ${STREAMLIT_PID} ${NGROK_PID:-} ${NGROK_OLLAMA_PID:-}"

echo "Logs: ngrok_streamlit.log (if created), ngrok_ollama.log (if created)"

# Bring script to foreground to keep child processes when run interactively
wait ${STREAMLIT_PID}
