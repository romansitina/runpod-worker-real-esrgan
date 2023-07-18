# Real-ESRGAN | RunPod Serverless Worker

The is the source code for a [RunPod](https://runpod.io?ref=w18gds2n)
Serverless worker that uses [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
for Restoration/Upscaling.

## Building the Worker

### Option 1: Network Volume

This will store your application on a Runpod Network Volume and
build a light weight Docker image that runs everything
from the Network volume without installing the application
inside the Docker image.

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
# Download the official Real-ESRGAN models
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth
# Download additional models from Huggingface
wget https://huggingface.co/snappic/upscalers/resolve/main/4x-UltraSharp.pth
wget https://huggingface.co/snappic/upscalers/resolve/main/lollypop.pth
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
10. Sign up for a Docker hub account if you don't already have one.
11. Build the Docker image and push to Docker hub:
```bash
docker build -t dockerhub-username/runpod-worker-real-esrgan:1.0.0 -f Dockerfile.Network_Volume .
docker login
docker push dockerhub-username/runpod-worker-real-esrgan:1.0.0
```

### Option 2: Standalone

This is the simpler option.  No network volume is required.
The entire application will be stored within the Docker image
but will obviously create a more bulky Docker image as a result.

```bash
docker build -t dockerhub-username/runpod-worker-real-esrgan:1.0.0 -f Dockerfile.Standalone .
docker login
docker push dockerhub-username/runpod-worker-real-esrgan:1.0.0
```

## Dockerfile

There are 2 different Dockerfile configurations

1. Network_Volume - See Option 1 Above.
2. Standalone - See Option 2 Above (No Network Volume is required for this option).

The worker is built using one of the two Dockerfile configurations
depending on your specific requirements.

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

## Acknowledgements

- [Real-ESRGAN (ai-forever)](https://github.com/ai-forever/Real-ESRGAN)

## Community and Contributing

Pull requests and issues on [GitHub](https://github.com/ashleykleynhans/runpod-worker-real-esrgan)
are welcome. Bug fixes and new features are encouraged.

You can contact me and get help with deploying your Serverless
worker to RunPod on the RunPod Discord Server below,
my username is **ashleyk**.

<a target="_blank" href="https://discord.gg/pJ3P2DbUUq">![Discord Banner 2](https://discordapp.com/api/guilds/912829806415085598/widget.png?style=banner2)</a>

## Appreciate my work?

<a href="https://www.buymeacoffee.com/ashleyk" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" ></a>
