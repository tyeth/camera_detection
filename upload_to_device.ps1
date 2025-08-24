# PowerShell script to generate SSH keys and upload files to Qualcomm device

# Variables
$DEVICE_IP = "192.168.1.169"
$DEVICE_USER = "root"
$REMOTE_PATH = "/home/particle/camera_detection"
$SSH_KEY_NAME = "id_rsa_qualcomm"
$SSH_KEY_PATH = "$env:USERPROFILE\.ssh\$SSH_KEY_NAME"

# Check if ssh-keygen is available
try {
    ssh-keygen -h | Out-Null
} catch {
    Write-Host "Error: ssh-keygen command not found. Make sure OpenSSH is installed."
    Write-Host "You can install it via 'Add Optional Feature' in Windows Settings or use Git Bash."
    exit 1
}

# Create .ssh directory if it doesn't exist
if (-not (Test-Path "$env:USERPROFILE\.ssh")) {
    New-Item -ItemType Directory -Path "$env:USERPROFILE\.ssh" | Out-Null
}

# Create SSH key if it doesn't exist
if (-not (Test-Path $SSH_KEY_PATH)) {
    Write-Host "Generating SSH key..."
    ssh-keygen -t rsa -b 4096 -f $SSH_KEY_PATH -N '""'
}

# Copy SSH public key to device (will prompt for password)
Write-Host "Uploading SSH key to device... (You'll be prompted for the device password)"
Write-Host "Running: ssh-copy-id -i $SSH_KEY_PATH.pub $DEVICE_USER@$DEVICE_IP"
Write-Host ""
Write-Host "Note: If 'ssh-copy-id' is not available on your system, manually run this command instead:"
Write-Host "type $SSH_KEY_PATH.pub | ssh $DEVICE_USER@$DEVICE_IP 'mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys'"
Write-Host ""

# Try ssh-copy-id, but provide alternative if not available
try {
    ssh-copy-id -i "$SSH_KEY_PATH.pub" "$DEVICE_USER@$DEVICE_IP"
} catch {
    $confirmation = Read-Host "Would you like to run the manual command instead? (y/n)"
    if ($confirmation -eq 'y') {
        Get-Content "$SSH_KEY_PATH.pub" | ssh "$DEVICE_USER@$DEVICE_IP" "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
    } else {
        Write-Host "Please manually add the SSH key to the device."
    }
}

# Create remote directory if it doesn't exist
Write-Host "Creating remote directory..."
ssh -i $SSH_KEY_PATH "$DEVICE_USER@$DEVICE_IP" "mkdir -p $REMOTE_PATH"

# Upload files to device
Write-Host "Uploading files to device..."
scp -i $SSH_KEY_PATH -r ./* "$DEVICE_USER@$DEVICE_IP`:$REMOTE_PATH/"

Write-Host "Done! You can now SSH into the device using:"
Write-Host "ssh -i $SSH_KEY_PATH $DEVICE_USER@$DEVICE_IP"
