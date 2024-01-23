import intel_extension_for_pytorch as ipex
import torch
import os
from diffusers import DiffusionPipeline, AutoencoderTiny

print(ipex.xpu.get_device_name(0))
pipe = DiffusionPipeline.from_pretrained("./LCM_Dreamshaper_v7", torch_dtype=torch.float16)
pipe.vae = AutoencoderTiny.from_pretrained("./LCM_Dreamshaper_v7/taesd", torch_dtype=torch.float16)
pipe = pipe.to("xpu")

prompt = "a majestic lion jumping over a stone at night"
for x in range(10):
    output = pipe(prompt = prompt, height=512, width=512, guidance_scale=1, num_inference_steps=4, num_images_per_prompt=1, output_type="pil").images[0]
    output.save("result"+ str(x) +'.png')
