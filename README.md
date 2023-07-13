# Real-ESRGAN | RunPod Serverless Worker

The is the source code for a [RunPod](https://runpod.io?ref=w18gds2n)
worker that uses [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
for Restoration/Upscaling.

## Building the Worker

#### Note: This worker requires a RunPod Network Volume this Python code preinstalled in order to function correctly.

### Network Volume

1. [Create a RunPod Account](https://runpod.io?ref=w18gds2n).
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
mkdir -p /workspace/ESRGAN/models
cd /workspace/ESRGAN/models
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth
mkdir -p /workspace/GFPGAN/models
wget -O /workspace/GFPGAN/models/GFPGANv1.3.pth https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth

# Install the worker application code and dependencies
cd /workspace
git clone https://github.com/ashleykleynhans/runpod-worker-real-esrgan.git
cd runpod-worker-real-esrgan
python3 -m venv venv
source venv/bin/activate
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

### Dockerfile

The worker is built using a Dockerfile. The Dockerfile specifies the
base image, environment variables, and system package dependencies.

## Running the Worker

The worker can be run using the `start.sh` script. This script starts the
init system and runs the serverless handler script.

## API

The worker provides an API for inference. The API payload looks like this:

```json
{
  "input": {
     "source_image": "base64 encoded source image content",
     "model": "RealESRGAN_x4plus",
     "scale": 2,
     "face_enhance": true
  }
}
```

The following models are available by default:

* RealESRGAN_x2plus
* RealESRGAN_x4plus
* RealESRNet_x4plus
* RealESRGAN_x4plus_anime_6B

## Serverless Handler

The serverless handler (`rp_handler.py`) is a Python script that handles
inference requests.  It defines a function handler(event) that takes an
inference request, runs the inference using the Real-ESRGAN model, and
returns the output as a JSON response in the following format:

```json
{
  "output": {
    "status": "ok",
    "image": "base64 encoded output image"
  }
}
```
