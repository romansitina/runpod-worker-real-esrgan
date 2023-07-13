#!/usr/bin/env python3
import io
import uuid
import base64
import json
import requests
import time
from PIL import Image

RUNPOD_API_KEY = 'INSERT_RUNPOD_API_KEY_HERE'
SERVERLESS_ENDPOINT_ID = 'INSERT_RUNPOD_ENDPOINT_ID_HERE'
RUNPOD_ENDPOINT_BASE_URL = f'https://api.runpod.ai/v2/{SERVERLESS_ENDPOINT_ID}'
SOURCE_IMAGE = 'data/src.jpg'


def encode_image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()
        encoded_data = base64.b64encode(image_data).decode('utf-8')
        return encoded_data


def save_result_image(resp_json):
    if resp_json['output']['image'] is None:
        raise Exception('No image received from RunPod endpoint')

    img = Image.open(io.BytesIO(base64.b64decode(resp_json['output']['image'])))
    output_file = f'{uuid.uuid4()}.jpg'

    with open(output_file, 'wb') as f:
        print(f'Saving image: {output_file}')
        img.save(f, format='JPEG')


if __name__ == '__main__':
    # Load the image and encode it to base64
    source_image_base64 = encode_image_to_base64(SOURCE_IMAGE)

    # Create the payload dictionary
    payload = {
        "input": {
            "source_image": source_image_base64,
            "model": "RealESRGAN_x4plus",
            "scale": 2,
            "face_enhance": True,
        }
    }

    r = requests.post(
        f'{RUNPOD_ENDPOINT_BASE_URL}/runsync',
        headers={
            'Authorization': f'Bearer {RUNPOD_API_KEY}'
        },
        json=payload
    )

    print(f'Status code: {r.status_code}')

    if r.status_code == 200:
        resp_json = r.json()

        if 'output' in resp_json and 'image' in resp_json['output']:
            save_result_image(resp_json)
        else:
            job_status = resp_json['status']
            print(f'Job status: {job_status}')

            if job_status == 'IN_QUEUE' or job_status == 'IN_PROGRESS':
                request_id = resp_json['id']
                request_in_queue = True

                while request_in_queue:
                    r = requests.get(
                        f'{RUNPOD_ENDPOINT_BASE_URL}/status/{request_id}',
                        headers={
                            'Authorization': f'Bearer {RUNPOD_API_KEY}'
                        }
                    )

                    print(f'Status code from RunPod status endpoint: {r.status_code}')

                    if r.status_code == 200:
                        resp_json = r.json()
                        job_status = resp_json['status']

                        if job_status == 'IN_QUEUE' or job_status == 'IN_PROGRESS':
                            print(f'RunPod request {request_id} is {job_status}, sleeping for 5 seconds...')
                            time.sleep(5)
                        elif job_status == 'FAILED':
                            request_in_queue = False
                            print(f'RunPod request {request_id} failed')
                        elif job_status == 'COMPLETED':
                            request_in_queue = False
                            print(f'RunPod request {request_id} completed')
                            save_result_image(resp_json)
                        elif job_status == 'TIMED_OUT':
                            request_in_queue = False
                            print(f'ERROR: RunPod request {request_id} timed out')
                        else:
                            request_in_queue = False
                            print(f'ERROR: Invalid status response from RunPod status endpoint')
                            print(json.dumps(resp_json, indent=4, default=str))
            elif job_status == 'COMPLETED' and resp_json['output']['status'] == 'error':
                print(f'ERROR: {resp_json["output"]["message"]}')
            else:
                print(json.dumps(resp_json, indent=4, default=str))
    else:
        print(f'ERROR: {r.content}')
