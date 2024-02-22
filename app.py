import tkinter as tk

from PIL import ImageTk
from hug_token import auth_token

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

app = tk.Tk()
app.geometry("600x700")
app.title("🎨 Creative Image Generator 🖼️")

def gen():
        with autocast(gpu):
            image = pipeline(prompt.get(), guidance_scale=8.5)["sample"][0]
        image.save("genimage.png")
        img = ImageTk.PhotoImage(image)
        frame.configure(image=img)

prompt = tk.Entry(app, width=50, font=("Arial", 16))
prompt.place(x=30, y=20)

frame = tk.Label(app, height=400, width=500)
frame.place(x=50, y=100)

button = tk.Button(app, text="Generate", width=20, font=("Arial", 16), bg="orange", fg="white", command=gen)
button.place(x=220, y=60)

gpu = "cuda"
modelid = "CompVis/stable-diffusion-v1-4"
pipeline = StableDiffusionPipeline.from_pretrained(modelid, variant="fp16", torch_dtype=torch.float16, use_token=auth_token)
pipeline.to(gpu)

app.mainloop()
