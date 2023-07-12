#!/usr/bin/env python3
import base64
import json

SOURCE_IMAGE = 'src_upscale.jpeg'


def encode_image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()
        encoded_data = base64.b64encode(image_data).decode('utf-8')
        return encoded_data


if __name__ == '__main__':
    source_image_base64 = encode_image_to_base64(SOURCE_IMAGE)

    # Create the payload dictionary
    payload = {
        "input": {
            "source_image": source_image_base64,
            "model": "RealESRGAN_x4plus",
            "scale": 2,
            "face_enhance": True
        }
    }

    # Save the payload to a JSON file
    with open('test_input.json', 'w') as output_file:
        json.dump(payload, output_file)

    print('Payload saved to: test_input.json')

