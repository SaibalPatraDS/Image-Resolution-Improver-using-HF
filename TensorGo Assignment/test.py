## Huggingface Implementation
import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline
import torch
from custom_file import UpScalingPipeline


## Load the model and schedular
model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_type = torch.float16)
pipeline = pipeline.to('cuda')



## Get Components from UPScaler Pipeline
vae = pipeline.vae
tokenizer = pipeline.tokenizer
text_encoder = pipeline.text_encoder
unet = pipeline.unet
low_res_scheduler = pipeline.low_res_scheduler
scheduler = pipeline.scheduler
image_processor = pipeline.image_processor

## Final Pipe
pipe = UpScalingPipeline(vae, tokenizer, text_encoder, unet, low_res_scheduler, 
                         scheduler, image_processor)

# print(pipe)



## Testing the Pipeline
img_path = "6X4H.jpg"
image = Image.open(img_path).convert('RGB')
prompt = "Make sure to prserve the context of the image"
upscaled_img = pipe(prompt=prompt, image=image)[0]
upscaled_img