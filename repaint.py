import torch
import torchvision
from torchvision import transforms
from diffusers import UNet2DModel
from torch.optim import Adam
import tqdm
from PIL import Image


def get_model():
    block_out_channels=(128, 128, 256, 256, 512, 512)
    down_block_types=( 
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D", 
        "DownBlock2D", 
        "DownBlock2D", 
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    )
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D"  
    )
    return UNet2DModel(block_out_channels=block_out_channels,out_channels=3, in_channels=3, up_block_types=up_block_types, down_block_types=down_block_types, add_attention=True)

# Go from x_0 to x_1, x_1 is real data distribution, x_0 is prior

# Forward diffuse from x_1 to x_t,

def decrease_alpha(x_alpha2, alpha1, alpha2):

    z = torch.randn(*x_alpha2.shape, device=device)

    l = torch.sqrt((1-alpha1)**2 - (alpha1/alpha2*(1-alpha2))**2)

    x_alpha1 = (alpha1/alpha2)[:,None,None,None] * x_alpha2 + l[:,None,None,None] * z

    return x_alpha1

@torch.no_grad()
def inpaint(model, x0, x1, mask, nb_step, U=5):
    x_alpha = x0

    for t in tqdm.tqdm(range(nb_step)):
        for u in range(1, U + 1):
            alpha_start = torch.FloatTensor([(t/nb_step)]).cuda()
            alpha_end = torch.FloatTensor([((t+1)/nb_step)]).cuda()

            
            d = model(x_alpha, torch.tensor(alpha_start, device=x_alpha.device))['sample']
            x_alpha_forward = x_alpha + (alpha_end-alpha_start)*d
            
            x_alpha_backward = (1-alpha_end) * x0 + (alpha_end) * x1
            
            x_alpha = mask * x_alpha_forward + (1-mask) * x_alpha_backward
            if u < U and t < nb_step - 1:
                x_alpha = decrease_alpha(x_alpha, alpha_start, alpha_end)
    return x_alpha

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CELEBA_FOLDER = './datasets/celeba/'

transform = transforms.Compose([transforms.Resize(64),transforms.CenterCrop(64), transforms.RandomHorizontalFlip(0.5),transforms.ToTensor()])
train_dataset = torchvision.datasets.CelebA(root=CELEBA_FOLDER, split='train',
                                        download=True, transform=transform)

batch_size=8
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

model = get_model()
model = model.to(device)
model.load_state_dict(torch.load("celeba.ckpt"))

data = next(iter(dataloader))

x1 = (data[0].to(device)*2)-1
x0 = torch.randn_like(x1)
bs = x0.shape[0]

mask = torch.zeros_like(x0)
mask[:, :, 32-10:32+20, 32-10:32+10] = 1


for U in range(1, 10):
    sample = (inpaint(model, x0, x1, mask, nb_step=128, U=U) * 0.5) + 0.5

    bool_mask = mask.type(torch.bool)
    masked_x1 = x1.clone()
    masked_x1[bool_mask] = 0

    plot = torch.cat(((x1 + 1)/2, (masked_x1 + 1)/2, sample), dim=0)

    grid = torchvision.utils.make_grid(plot, nrow=batch_size)

    torchvision.utils.save_image(grid, f'test_{U}.png')