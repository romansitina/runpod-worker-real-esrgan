# Real-ESRGAN | RunPod Worker

The is the source code for a RunPod worker that uses
[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
for Restoration/Upscaling.

## Building the Worker

#### Note: This worker requires a RunPod Network Volume with Real-ESRGAN preinstalled in order to function correctly.

### Network `Volume`

1. Create a [RunPod Network Volume](https://www.runpod.io/console/user/storage).
2. Attach the Network Volume to a Secure Cloud [GPU pod](https://www.runpod.io/console/gpu-secure-cloud).
3. Select a light-weight template such as RunPod Pytorch.
4. Deploy the GPU Cloud pod.
5. Once the pod is up, open a Terminal and install inswapper:
```bash
cd /workspace
git clone https://github.com/xinntao/Real-ESRGAN.git
cd Real-ESRGAN
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
# TODO: Check rest of the instructions
```
6. Install the RunPod Python module which is required for the worker to function correctly within RunPod Serverless:
```bash
pip3 install runpod
```
7. Run the example inference so that the models can be cached on
   your Network Volume, which will dramatically reduce cold start times for RunPod Serverless:
# TODO: Fix
```bash
python3 swapper.py \
  --source_img /workspace/inswapper/data/src.jpg \
  --target_img /workspace/inswapper/data/target.jpg \
  --face_restore \
  --background_enhance \
  --face_upsample \
  --upscale 1 \
  --codeformer_fidelity 0.5
```

### Dockerfile

The worker is built using a Dockerfile. The Dockerfile specifies the
base image, environment variables, and system package dependencies

The Python dependencies are specified in requirements.txt.
The primary dependency is `runpod==0.10.0`.

## Running the Worker

The worker can be run using the start.sh script. This script starts the
init system and runs the serverless handler script.

## API

The worker provides an API for inference. The API payload looks like this:

```json
{
  "input": {
    "image": "base64 encoded source image content"
  }
}
```

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
