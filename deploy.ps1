# UTF-8 Encoding (Ensure this file is saved with UTF-8 with BOM if using ISE)
$SERVER_IP = "1.12.66.237"
$SERVER_USER = "root"
$REMOTE_PATH = "/root/strategy_backtest"

Write-Host ">>> Starting Deployment Process..." -ForegroundColor Cyan

# 1. Uploading files using SCP
# We use -r to copy the current directory content
Write-Host ">>> Step 1: Uploading code to $SERVER_IP..." -ForegroundColor Yellow
# SCP on Windows sometimes struggles with complex exclusions, so we sync the core files
scp -r ./* "${SERVER_USER}@${SERVER_IP}:${REMOTE_PATH}"

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Upload failed. Please check SSH connection." -ForegroundColor Red
    exit
}

# 2. Remote execution using SSH
# To avoid && parsing issues in PS, we wrap the remote command in a single string carefully
Write-Host ">>> Step 2: Rebuilding Docker containers on server..." -ForegroundColor Yellow
$remoteCmd = "cd ${REMOTE_PATH}; docker compose up -d --build"
ssh "${SERVER_USER}@${SERVER_IP}" $remoteCmd

if ($LASTEXITCODE -eq 0) {
    Write-Host "SUCCESS: Deployment completed!" -ForegroundColor Green
    Write-Host "URL: http://${SERVER_IP}:8501" -ForegroundColor Cyan
} else {
    Write-Host "ERROR: Remote execution failed. Check Docker logs on server." -ForegroundColor Red
}
