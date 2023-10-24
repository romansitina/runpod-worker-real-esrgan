#!/usr/bin/env bash

echo "Worker Initiated"

echo "Symlinking files from Network Volume"
ln -s /runpod-volume /workspace
rm -rf /root/.cache
ln -s /runpod-volume/.cache /root/.cache

echo "Starting RunPod Handler"
export PYTHONUNBUFFERED=1
cd /workspace/runpod-worker-real-esrgan
python3 -u rp_handler.py
