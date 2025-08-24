#!/bin/bash

# Script to generate SSH keys and upload files to Qualcomm device

# Variables
DEVICE_IP="192.168.1.169"
DEVICE_USER="root"
REMOTE_PATH="/home/particle/camera_detection"
SSH_KEY_NAME="id_rsa_qualcomm"

# Create SSH key if it doesn't exist
if [ ! -f ~/.ssh/$SSH_KEY_NAME ]; then
    echo "Generating SSH key..."
    ssh-keygen -t rsa -b 4096 -f ~/.ssh/$SSH_KEY_NAME -N ""
fi

# Copy SSH public key to device (will prompt for password)
echo "Uploading SSH key to device..."
ssh-copy-id -i ~/.ssh/$SSH_KEY_NAME.pub $DEVICE_USER@$DEVICE_IP

# Create remote directory if it doesn't exist
echo "Creating remote directory..."
ssh -i ~/.ssh/$SSH_KEY_NAME $DEVICE_USER@$DEVICE_IP "mkdir -p $REMOTE_PATH"

# Upload files to device
echo "Uploading files to device..."
scp -i ~/.ssh/$SSH_KEY_NAME -r ./* $DEVICE_USER@$DEVICE_IP:$REMOTE_PATH/

echo "Done! You can now SSH into the device using:"
echo "ssh -i ~/.ssh/$SSH_KEY_NAME $DEVICE_USER@$DEVICE_IP"
