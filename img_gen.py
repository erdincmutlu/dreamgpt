import torch

from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline


def check_cuda_device()->torch.device:
    """Check if cuda is available and return the device"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def get_the_model(model_id:str = "stabilityai/stable-diffusion-2")->StableDiffusionPipeline:
    """Get the model"""
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    device = check_cuda_device()
    pipe.to(device)

    return pipe


def get_image_to_image_model(model_id:str="stabilityai/stable-diffusion-2")->StableDiffusionImg2ImgPipeline:
    """Get the image to image model"""
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    )

    device = check_cuda_device()
    pipe.to(device)

    return pipe


def gen_initial_img(int_prompt:str)->torch.Tensor:
    """Generate the initial image"""
    model = get_the_model()
    image = model(int_prompt, num_inference_steps=100).images[0]

    return image


def generate_story_images(int_prompt:str, story_steps:list)->dict:
    """Generate the ilustrations for the story"""
    image_dic = {}
    step_zero_img = gen_initial_img(int_prompt)
    img2img_model = get_image_to_image_model()

    initialisation_img = step_zero_img

    for idx, story_step in enumerate(story_steps):
        step_img = img2img_model(
            prompt=story_step,
            image=initialisation_img,
            strength=0.5,
            guidance_scale=6,
            num_inference_steps=100,
        ).images[0]
        image_dic[idx] = {"image": step_img, "prompt": story_step}
        initialisation_img = step_img

    return image_dic
