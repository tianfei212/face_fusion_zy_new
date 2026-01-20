# AI Face Swap Integrated App

This is a desktop application integrating the frontend and backend of the AI Face Swap project.

## Structure
- `frontend/`: React frontend (based on original web frontend)
- `backend/`: Python backend (FastAPI/ZMQ)
- `main.js`: Electron main process

## Prerequisites
- Node.js & npm
- Python 3.10+ with dependencies installed
- NVIDIA GPU with CUDA support (for backend)

## Setup & Run

1. Build Frontend:
   ```bash
   cd frontend
   npm install
   npm run build
   cd ..
   ```

2. Run Application:
   ```bash
   npm install
   npm start
   ```

## Notes
- The application spawns the Python backend automatically using `python3 backend/main.py`.
- Ensure your Python environment has all necessary libraries installed (torch, onnxruntime-gpu, fastapi, etc.).
