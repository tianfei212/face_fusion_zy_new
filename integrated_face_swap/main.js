const { app, BrowserWindow } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow;
let backendProcess;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 720,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    }
  });

  // Load the built frontend
  const indexPath = path.join(__dirname, 'frontend/dist/index.html');
  console.log('Loading frontend from:', indexPath);
  mainWindow.loadFile(indexPath);

  // Open the DevTools.
  // mainWindow.webContents.openDevTools();

  mainWindow.on('closed', function () {
    mainWindow = null;
  });
}

function startBackend() {
  const backendPath = path.join(__dirname, 'backend');
  console.log('Starting backend from:', backendPath);
  
  // Using python3, assuming dependencies are installed in the environment
  backendProcess = spawn('python3', ['main.py'], {
    cwd: backendPath,
    stdio: 'inherit'
  });

  backendProcess.on('error', (err) => {
    console.error('Failed to start backend:', err);
  });

  backendProcess.on('exit', (code, signal) => {
    console.log(`Backend process exited with code ${code} and signal ${signal}`);
  });
}

app.on('ready', () => {
  startBackend();
  // Wait a bit for backend to initialize before showing window
  setTimeout(createWindow, 2000);
});

app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', function () {
  if (mainWindow === null) createWindow();
});

app.on('will-quit', () => {
  if (backendProcess) {
    console.log('Killing backend process...');
    backendProcess.kill();
  }
});
