import torch
import Model
import ForwardProcess as FP
import ReverseProcess as RP
import numpy as np
from PIL import Image,ImageTk
import os
import tkinter as tk
from tkinter import simpledialog
from ClipConditioning import ClipConditioning

# 使用tkinter创建一个输入对话框
root = tk.Tk()
root.withdraw()  # 隐藏主窗口

text_condition = simpledialog.askstring(title="Text Input",
                                        prompt="Enter text condition for the model:")
root.destroy()

device = "cuda" if torch.cuda.is_available() else "cpu"

image_size = 64
channels = 3
model = Model.Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2, 4,)
)
model.to(device)

model.load_state_dict(torch.load('G:/Python/Miyazaki/Miyazaki_Stable Diffusion/state/unet_model.pt'))
model.eval()

clip_conditioning = ClipConditioning(device=device)
text_condition = "a horse"
condition = clip_conditioning.get_text_condition(text_condition)
print("Text Condition:", text_condition)

elements = FP.GetElements()
samples = RP.sample(model,
                    image_size=image_size,
                    batch_size=32,
                    channels=channels,
                    betas=elements[0],
                    sqrt_recip_alphas=elements[4],
                    sqrt_one_minus_alphas_cumprod=elements[6],
                    posterior_variance=elements[7],
                    condition=condition)

print("shape of samples:", np.shape(samples))
samples = torch.tensor(samples, dtype=torch.float32)

if not os.path.exists('G:/Python/Miyazaki/Miyazaki_Stable Diffusion/result'):
    os.makedirs('G:/Python/Miyazaki/Miyazaki_Stable Diffusion/result')

last_image = None

for i in range(samples.shape[0]):
    image_index = 25
    image = samples[i, image_index]

    img_normalized = ((image - image.min()) * (255 / (image.max() - image.min())))

    img_normalized = img_normalized.numpy().astype(np.uint8)
    img_normalized = np.transpose(img_normalized, (1, 2, 0))
    img_pil = Image.fromarray(img_normalized, 'RGB')

    if i % 10 == 0:     # 每10次去噪保存一次图片
        img_pil.save(f'G:/Python/Miyazaki/Miyazaki_Stable Diffusion/result/time_step_{i}.png')

    last_image = img_pil

if last_image is not None:
    root = tk.Tk()
    root.title("Generated Image")

    img_display = ImageTk.PhotoImage(last_image)
    panel = tk.Label(root,image=img_display)
    panel.pack(side="bottom",fill="both",expand=True)

    root.mainloop()

print("ending.")