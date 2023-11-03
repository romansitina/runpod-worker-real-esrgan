#!/usr/bin/env python3
from util import post_request, encode_image_to_base64

SOURCE_IMAGE = '../data/src_upscale.jpeg'


if __name__ == '__main__':
    source_image_base64 = encode_image_to_base64(SOURCE_IMAGE)

    payload = {
        "input": {
            "source_image": source_image_base64,
            "model": "RealESRGAN_x4plus",
            "scale": 2,
            "face_enhance": True,
            "half": False
        }
    }

    post_request(payload)
