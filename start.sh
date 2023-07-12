#!/usr/bin/env bash

echo "Worker Initiated"

echo "Symlinking files from Network Volume"
ln -s /runpod-volume /workspace
rm -rf /root/.cache
ln -s /runpod-volume/.cache /root/.cache

echo "Starting RunPod Handler"
source /workspace/Real-ESRGAN/venv/bin/activate
mv /rp_handler.py /workspace/Real-ESRGAN/rp_handler.py
cd /workspace/Real-ESRGAN
python -u rp_handler.py
