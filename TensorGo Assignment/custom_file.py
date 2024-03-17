## Custom Implementation of Low Resolution Image to High Resolution Image

from tqdm import tqdm
from torch import autocast
import torch

## Python Class to upscale Image

class UpScalingPipeline:
    '''Custom Implementation of the Stable Diffusion Upscalling Pipeline'''
    def __init__(self, vae, tokenizer, text_encoder, unet, 
                 low_res_scheduler, scheduler, image_processor):
        self.vae = vae
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.unet = unet
        self.low_res_scheduler = low_res_scheduler
        self.scheduler = scheduler
        self.image_processor = image_processor
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ## Text Embeddings for Prompt
    def get_text_embeddings(self, text):
        """Returns Embeddings of the given text"""
        #Tokenize the Text
        text_input = self.tokenizer(text, padding = 'max_length', max_length = self.tokenizer.model_max_length,
                                    truncation = True, return_tensors = 'pt')
        # embed the text
        with torch.no_grad():
            text_embeds = self.text_encoder(text_input.input_ids.to(self.device))

        return text_embeds
    
    # Get the Prompt
    def get_prompt_embeds(self, prompt):
        """Return Prompts based on classifier free guidance"""
        if isinstance(prompt, str):
            prompt = [prompt]
        #get conditional prompt embeddings
        cond_outputs = self.get_text_embeddings(prompt)
        cond_embeds = cond_outputs.last_hidden_state
        # get uncoditional prompt embeddings
        uncond_outputs = self.get_text_embeddings([''] * len(prompt))
        uncond_embeds = uncond_outputs.last_hidden_state
        # concatenate the conditional and unconditional embeddings for classifier free guidance
        prompt_embeds = torch.cat([uncond_embeds, cond_embeds])
        return prompt_embeds

    ## Method for handling all Image preprocessing
    def transform_image(self, image):
        """Convert image from pytorch tensor to PIL Format"""
        return self.image_processor.postprocess(image, output_type = 'pil')
    
    ## Method to get Initial Image Latents to be denoised in the denoising step
    def get_initial_latents(self, height, weight, num_channels_latents, batch_size):
        """Returns noise latent tensor of revelant shape scaled by the scheduler"""
        image_latents = torch.randn((batch_size, num_channels_latents, height, weight)).to(self.device)
        ## Scaling Initial Noise by the standard deviation required by the scheduler
        image_latents = image_latents * self.scheduler.init_noise_sigma
        return image_latents
    
    ## Defining Denoising Method
    def denoise_latents(self, prompt_embeds, image, timesteps, latents, noise_level, guidance_scale):
        """Denoises latents from noisy latents to meaningful latents"""
        with autocast('cuda'):
            for i, t in tqdm(enumerate(timesteps)):
                ## duplicate image latents for classifier free guidance
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = torch.cat([latent_model_input, image], dim = 1)

                ## Predict Noise Residuals
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input,
                                           t,
                                           encoder_hidden_states = prompt_embeds,
                                           class_labels = noise_level)['sample']
                    # seperate predictions for unconditional and conditional outputs
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    # perform guidance
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    # remove noise from the current sample
                    latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

        return latents

    ## Final ready to use methods
    def __call__(self, prompt, image, num_inference_steps = 20, guidance_scale = 9.0,
                 noise_level = 20):
        """Generate New Image Based on the prompt and the Image"""
        ## encode input image
        prompt_embeds = self.get_prompt_embeds(prompt)
        ## Preprocess Image
        image = self.image_processor.preprocess(image).to(self.device)
        ## prepare timestamps
        self.scheduler.set_timesteps(num_inference_steps, device = self.device)
        timesteps = self.scheduler.timesteps
        ## Add noise to the image
        noise_level = torch.tensor([noise_level], device = self.device)
        noise = torch.randn(image.shape, device = self.device)
        image = self.low_res_scheduler.add_noise(image, noise, noise_level)
        ## duplicate image for classifier free guidance
        image = torch.cat([image] * 2)
        noise_level = torch.cat([noise_level] * image.shape[0])
        ## prepare the initial image in the latent space(noise on which reverse diffusion will be done)
        num_channels_latents = self.vae.config.latent_channels
        batch_size = prompt_embeds.shape[0] // 2
        height, width = image.shape[2:]
        latents = self.get_initial_latents(height, width, num_channels_latents, batch_size)
        ## Denoise Latents
        latents = self.denoise_latents(prompt_embeds, image, 
                                       timesteps, latents, noise_level, guidance_scale)
        ## Decode Latents to get the image into pixel space
        latents = latents.to(torch.float16)
        image = self.vae.decode(latents/self.vae.config.scaling_factor, return_dict = False)[0]
        return image




