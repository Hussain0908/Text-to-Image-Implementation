import tkinter as tk
import customtkinter as ctk

from PIL import ImageTk
from hug_token import auth_token

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

app = tk.Tk()
app.geometry("532x632")
app.title("Image Generation")
ctk.set_appearance_mode("dark")

prompt = ctk.CTkEntry(master=None, height=40, width=512, font=("Arial", 20), text_color="black", fg_color="white")
prompt.place(x=10, y=10)

frame = ctk.CTkLabel(master=None, height=512, width=512)
frame.place(x=10, y=110)

gpu = "cuda"
modelid = "CompVis/stable-diffusion-v1-4"
pipeline = StableDiffusionPipeline.from_pretrained(modelid, variant="fp16", torch_dtype=torch.float16, use_token=auth_token)
pipeline.to(gpu)

def gen():
    with autocast(gpu):
        image = pipeline(prompt.get(), guidance_scale=8.5)["sample"][0]

    image.save("genimage.png")
    img = ImageTk.PhotoImage(image)
    frame.configure(image=img)

button = ctk.CTkButton(master=None, height=20, width=120, font=("Arial", 20), text_color="white", fg_color="red", command=gen)
button.configure(text="Generate")
button.place(x=206, y=60)


app.mainloop()
