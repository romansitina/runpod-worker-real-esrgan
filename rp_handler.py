import os
import io
import uuid
import base64
import cv2
import glob
import runpod
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
# from realesrgan.archs.srvgg_arch import SRVGGNetCompac
from PIL import Image

GPU_ID = 0
TMP_PATH = '/tmp/upscaler'
MODELS_PATH = '/workspace/ESRGAN/models'
GFPGAN_MODEL_PATH = '/workspace/GFPGAN/models/GFPGANv1.3.pth'
script_dir = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------- #
# Application Functions                                                        #
# ---------------------------------------------------------------------------- #
def upscale(
        source_image_path,
        image_extension,
        model_name='RealESRGAN_x4plus',
        outscale=4,
        face_enhance=False,
        fp32=False,
        tile=0,
        tile_pad=10,
        pre_pad=0,
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

    face_enhance:

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
        half=not fp32,
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

    try:
        if face_enhance:
            _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        else:
            output, _ = upsampler.enhance(img, outscale=outscale)
    except RuntimeError as error:
        print('Error', error)
        print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
    else:
        result_image = Image.fromarray(output)
        output_buffer = io.BytesIO()
        result_image.save(output_buffer, format='JPEG')
        image_data = output_buffer.getvalue()
        return base64.b64encode(image_data).decode('utf-8')


def determine_file_extension(image_data):
    image_extension = None
    # You can add more checks for other image formats if necessary
    if image_data.startswith('/9j/'):
        image_extension = '.jpg'
    elif image_data.startswith('iVBORw0Kg'):
        image_extension = '.png'
    # Add more image format checks as needed

    if image_extension is None:
        raise ValueError('Invalid image data')

    return image_extension


def upscaling_api(input):
    if not os.path.exists(TMP_PATH):
        os.makedirs(TMP_PATH)

    if 'source_image' not in input:
        raise Exception('Invalid payload')

    unique_id = uuid.uuid4()
    source_image_data = input['source_image']
    model_name = input['model']
    outscale = input['scale']
    face_enhance = input['face_enhance']

    # Decode the source image data
    source_image = base64.b64decode(source_image_data)
    source_file_extension = determine_file_extension(source_image_data)
    source_image_path = f'{TMP_PATH}/source_{unique_id}{source_file_extension}'

    # Save the source image to disk
    with open(source_image_path, 'wb') as source_file:
        source_file.write(source_image)

    try:
        result_image = upscale(
            source_image_path,
            source_file_extension,
            model_name,
            outscale,
            face_enhance
        )
    except Exception as e:
        raise Exception('Upscale failed')

    # Clean up temporary images
    os.remove(source_image_path)

    return {
        'status': 'ok',
        'image': result_image
    }


# ---------------------------------------------------------------------------- #
# RunPod Handler                                                               #
# ---------------------------------------------------------------------------- #
def handler(event):
    '''
    This is the handler function that will be called by the serverless.
    '''

    return upscaling_api(event["input"])


if __name__ == "__main__":
    print("Starting RunPod Serverless...")
    runpod.serverless.start(
        {
            'handler': handler
        }
    )