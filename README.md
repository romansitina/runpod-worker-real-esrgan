# Real-ESRGAN | RunPod Serverless Worker

This repo is forked from https://github.com/ashleykleynhans/runpod-worker-real-esrgan

In this version instead of base64 image input and output URL for GET and POST are passed as parameters to overcome the data size API limits.

Tested with S3 pre-signed URLs.

It also contains deployment to Docker Hub using GitHub pipelines.  

--- 

## TODO
- explore https://github.com/Alhasan-Abdellatif/Real-ESRGAN-lp for seamless stitching

---

This is the source code for a [RunPod](https://runpod.io?ref=2xxro4sy)
Serverless worker that uses [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
for Restoration/Upscaling.

## Model

The following models are available by default:

* RealESRGAN_x2plus
* RealESRGAN_x4plus
* RealESRNet_x4plus
* RealESRGAN_x4plus_anime_6B

## Testing

1. [Local Testing](docs/testing/local.md)
2. [RunPod Testing](docs/testing/runpod.md)

## Building the Docker image that will be used by the Serverless Worker

There are two options:

1. [Network Volume](docs/building/with-network-volume.md)
2. [Standalone](docs/building/without-network-volume.md) (without Network Volume)

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
