## Building the Worker with a Network Volume

This will store your application on a Runpod Network Volume and
build a light weight Docker image that runs everything
from the Network volume without installing the application
inside the Docker image.

1. [Create a RunPod Account](https://runpod.io?ref=2xxro4sy).
2. Create a [RunPod Network Volume](https://www.runpod.io/console/user/storage).
3. Attach the Network Volume to a Secure Cloud [GPU pod](https://www.runpod.io/console/gpu-secure-cloud).
4. Select a light-weight template such as RunPod Pytorch.
5. Deploy the GPU Cloud pod.
6. Once the pod is up, open a Terminal and install the required dependencies:
```bash
# Link the cache to /workdpace so the container disk does not run out of space
mv /root/.cache /workspace/.cache
ln -s /workspace/.cache /root/.cache

# Install the models
mkdir -p /workspace/models/ESRGAN
cd /workspace/models/ESRGAN
# Download the official Real-ESRGAN models
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth
# Download additional models from Huggingface
#wget https://huggingface.co/snappic/upscalers/resolve/main/4x-UltraSharp.pth
#wget https://huggingface.co/snappic/upscalers/resolve/main/lollypop.pth
mkdir -p /workspace/models/GFPGAN
# Download GFPGAN model
cd /workspace/models/GFPGAN
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth

# Install the worker application code and dependencies
cd /workspace
git clone https://github.com/ashleykleynhans/runpod-worker-real-esrgan.git
cd runpod-worker-real-esrgan
python3 -m venv venv
source venv/bin/activate
pip3 install --no-cache-dir torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install -r requirements.txt
python3 setup.py develop
```
7. Edit the `create_test_json.py` file and ensure that you set `SOURCE_IMAGE` to
   a valid image to upscale (you can upload the image to your pod using
   [runpodctl](https://github.com/runpod/runpodctl/releases)).
8. Create the `test_input.json` file by running the `create_test_json.py` script:
```bash
python3 create_test_json.py
```
9. Run an inference on the `test_input.json` input so that the models can be cached on
   your Network Volume, which will dramatically reduce cold start times for RunPod Serverless:
```bash
python3 -u rp_handler.py
```
10. Sign up for a Docker hub account if you don't already have one.
11. Build the Docker image and push to Docker hub:
```bash
docker build -t dockerhub-username/runpod-worker-real-esrgan:1.0.0 -f Dockerfile.Network_Volume .
docker login
docker push dockerhub-username/runpod-worker-real-esrgan:1.0.0
```
