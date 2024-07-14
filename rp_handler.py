import io
import uuid
import cv2
import requests
import os
import traceback
import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.modules.rp_logger import RunPodLogger
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from PIL import Image
from schemas.input import INPUT_SCHEMA

GPU_ID = 0
VOLUME_PATH = '/workspace'
TMP_PATH = f'{VOLUME_PATH}/tmp'
MODELS_PATH = f'{VOLUME_PATH}/models/ESRGAN'
GFPGAN_MODEL_PATH = f'{VOLUME_PATH}/models/GFPGAN/GFPGANv1.3.pth'
logger = RunPodLogger()


# ---------------------------------------------------------------------------- #
# Application Functions                                                        #
# ---------------------------------------------------------------------------- #
def upscale(
        source_image_path,
        image_extension,
        model_name='RealESRGAN_x4plus',
        outscale=4,
        face_enhance=False,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
        denoise_strength=0.5
):
    """
    model_name options:
        - RealESRGAN_x4plus
        - RealESRNet_x4plus
        - RealESRGAN_x4plus_anime_6B
        - RealESRGAN_x2plus
        - realesr-animevideov3
        - realesr-general-x4v3

    image_extension: .jpg or .png

    outscale: The final upsampling scale of the image

    face_enhance: Whether or not to enhance the face

    tile: Tile size, 0 for no tile during testing

    tile_pad: Tile padding (default = 10)

    pre_pad: Pre padding size at each border

    denoise_strength: 0 for weak denoise (keep noise)
                      1 for strong denoise ability
                      Only used for the realesr-general-x4v3 model
    """

    # determine models according to model names
    model_name = model_name.split('.')[0]

    if image_extension == '.jpg':
        image_format = 'JPEG'
    elif image_extension == '.png':
        image_format = 'PNG'
    else:
        raise ValueError(f'Unsupported image type, must be either JPEG or PNG')

    if model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
    elif model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
    # TODO: Implement these
    # elif model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
    #     model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
    #     netscale = 4
    #     file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
    # elif model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
    #     model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
    #     netscale = 4
    #     file_url = [
    #         'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
    #         'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
    #     ]
    elif model_name == '4x-UltraSharp':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif model_name == 'lollypop':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    else:
        raise ValueError(f'Unsupported model: {model_name}')

    # determine model paths
    model_path = os.path.join(MODELS_PATH, model_name + '.pth')

    if not os.path.isfile(model_path):
        raise Exception(f'Could not find model: {model_path}')

    # use dni to control the denoise strength
    dni_weight = None
    # if model_name == 'realesr-general-x4v3' and denoise_strength != 1:
    #     wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
    #     model_path = [model_path, wdn_model_path]
    #     dni_weight = [denoise_strength, 1 - denoise_strength]

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=half,
        gpu_id=GPU_ID
    )

    if face_enhance:  # Use GFPGAN for face enhancement
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path=GFPGAN_MODEL_PATH,
            upscale=outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler
        )

    img = cv2.imread(source_image_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise RuntimeError(f'Source image ({source_image_path}) is corrupt')

    try:
        if face_enhance:
            _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        else:
            output, _ = upsampler.enhance(img, outscale=outscale)
    except RuntimeError as e:
        logger.info("err", e)
        raise RuntimeError(e)
    else:
        result_image = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        output_buffer = io.BytesIO()
        result_image.save(output_buffer, format=image_format)
        image_data = output_buffer.getvalue()
        return image_data


def upscaling_api(input):
    if not os.path.exists(TMP_PATH):
        os.makedirs(TMP_PATH)

    unique_id = uuid.uuid4()
    source_url = input['source_url']
    output_url = input['output_url']
    model_name = input['model']
    outscale = input['scale']
    face_enhance = input['face_enhance']
    tile = input['tile']
    tile_pad = input['tile_pad']
    pre_pad = input['pre_pad']
    half = input['half']

    source_image_path, source_file_extension = download_file_from_presigned_url(source_url,
                                                                                f'{TMP_PATH}/source_{unique_id}')

    try:
        result_image = upscale(
            source_image_path,
            source_file_extension,
            model_name,
            outscale,
            face_enhance,
            tile,
            tile_pad,
            pre_pad,
            half
        )
    except Exception as e:
        logger.error(f'An exception was raised: {e}')

        return {
            'error': traceback.format_exc(),
            'refresh_worker': True
        }

    # Clean up temporary images
    os.remove(source_image_path)

    upload_file_to_presigned_url(output_url, result_image)

    return {
        'image': output_url
    }


def download_file_from_presigned_url(presigned_url, save_location):
    try:
        response = requests.get(presigned_url)
        response.raise_for_status()  # Raise an HTTPError if the HTTP request returned an unsuccessful status code

        # Extract the file extension from the URL
        file_extension = os.path.splitext(presigned_url.split('?')[0])[1]
        filename = os.path.basename(presigned_url.split('?')[0])

        # Save the file to the given location with the original extension
        file_path = os.path.join(save_location, filename)
        os.makedirs(save_location, exist_ok=True)
        with open(file_path, 'wb') as file:
            file.write(response.content)

        print(f"File downloaded and saved to {file_path}")
        return file_path, file_extension
    except requests.RequestException as e:
        print(f"An error occurred while trying to download the file: {e}")


def upload_file_to_presigned_url(presigned_url, file_bytestream):
    try:
        headers = {'Content-Type': 'application/octet-stream'}
        response = requests.put(presigned_url, data=file_bytestream, headers=headers)
        response.raise_for_status()  # Raise an HTTPError if the HTTP request returned an unsuccessful status code

        print("File uploaded successfully.")
    except requests.RequestException as e:
        print(f"An error occurred while trying to upload the file: {e}")


# ---------------------------------------------------------------------------- #
# RunPod Handler                                                               #
# ---------------------------------------------------------------------------- #
def handler(event):
    validated_input = validate(event['input'], INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {
            'errors': validated_input['errors']
        }

    return upscaling_api(validated_input['validated_input'])


if __name__ == "__main__":
    logger.info('Starting RunPod Serverless...')
    runpod.serverless.start(
        {
            'handler': handler
        }
    )
