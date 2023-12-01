## Building the Worker with a Network Volume

This is the simpler option.  No network volume is required.
The entire application will be stored within the Docker image
but will obviously create a more bulky Docker image as a result.

You can either use my pre-built Docker image:
```
ashleykza/runpod-worker-real-esrgan:standalone-1.1.4
```

Or alternatively, you can built it yourself following the
instructions below.

### Clone the repo

```bash
git clone https://github.com/ashleykleynhans/runpod-worker-real-esrgan.git
```

### Build the Docker image

```bash
docker build -t dockerhub-username/runpod-worker-real-esrgan:1.0.0 -f Dockerfile.Standalone .
docker login
docker push dockerhub-username/runpod-worker-real-esrgan:1.0.0
```
