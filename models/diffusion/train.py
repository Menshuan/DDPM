import torch
import Loss
import Model
import ForwardProcess as FP
from torch.optim import Adam
from torchvision.utils import save_image
import matplotlib.pyplot as plt

elements = FP.GetElements()

#            [betas,# β参数，控制噪声的加入;
#             alphas,
#             alphas_cumprod,
#             alphas_cumprod_prev,
#             sqrt_recip_alphas,  # α的平方根的倒数;
#             sqrt_alphas_cumprod,
#             sqrt_one_minus_alphas_cumprod,  # 1-α的累积乘积的平方根;
#             posterior_variance]


device = "cuda" if torch.cuda.is_available() else "cpu"

# 确定模型参数；
image_size = 64
channels = 3
model = Model.Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2, 4,)
)
model.to(device)
# 确定优化器；
optimizer = Adam(model.parameters(), lr=1e-5)
# 确定dataset;
dataset = Loss.makeDataLoader()

epochs = 1  # 训练 10 个 epoch
timesteps = 300
loss_values = []

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(loss_values)
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.set_title('Loss over Time')

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

save_and_sample_every = 1000

for epoch in range(epochs):
    for step, batch in enumerate(dataset):
        optimizer.zero_grad()
        batch_size = batch.shape[0]
        batch = batch.to(device)

        t = torch.randint(0, timesteps, (batch_size,), device=device).long()

        loss = Loss.p_losses(model,
                             batch,
                             t,
                             elements[5],
                             elements[6],
                             loss_type="huber",
                             use_perceptual_loss=True)

        loss_values.append(loss.item())

        if step % 100 == 0:
            print("Epoch", epoch, "Step", step, "Loss", loss.item())

        line.set_ydata(loss_values)
        line.set_xdata(range(len(loss_values)))
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), 'G:/Python/Miyazaki/Miyazaki_Stable Diffusion/state/unet_model.pt')  # 保存模型状态字典
torch.save(optimizer.state_dict(), 'G:/Python/Miyazaki/Miyazaki_Stable Diffusion/state/optimizer.pt')  # 保存优化器状态字典

plt.ioff()
plt.show()