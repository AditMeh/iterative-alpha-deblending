import torch
import torchvision
from torchvision import transforms
from diffusers import UNet2DModel
from torch.optim import Adam
import torchvision.datasets as datasets
from torch.distributions.uniform import Uniform
import torch.nn as nn
import wandb
import tqdm
import os 
import imageio 

T = 128
epochs = 20000
lr = 1e-4


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
    return UNet2DModel(block_out_channels=block_out_channels, out_channels=3, in_channels=3, up_block_types=up_block_types, down_block_types=down_block_types, add_attention=True)


def train():
    CELEBA_FOLDER = './datasets/celeba/'

    transform = transforms.Compose([transforms.Resize(size=(128, 128)), transforms.RandomHorizontalFlip(0.5),transforms.ToTensor()])
    train_dataset = torchvision.datasets.CelebA(root=CELEBA_FOLDER, split='train',
                                            download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, drop_last=True)
    
    model = get_model().cuda()
    model.load_state_dict(torch.load("celeba.pt"))

    optimizer = Adam(model.parameters(), lr=lr)

    # wandb.login()

    # run = wandb.init(
    #     # Set the project where this run will be logged
    #     project="iterative alpha deblending",
    #     # Track hyperparameters and run metadata
    #     config={
    #         "learning_rate": lr,
    #         "epochs": epochs,
    #     },
    #     name="celeba_256")

    
    for epoch in range(1, epochs+1):
        acc = 0
        denom = 0
        for x_1, _ in tqdm.tqdm(train_loader): 
            x_1 = x_1.cuda()
            x_0 = torch.randn(x_1.shape).cuda()

            # Broadcast to (B,c,h,w)
            alpha = Uniform(0, 1).sample((x_0.shape[0],)).cuda()
            alpha_broadcast = alpha[..., None, None, None]

            x_alpha = (1-alpha_broadcast) * x_0 + alpha_broadcast * x_1

            pred = model(x_alpha, alpha)['sample']

            loss = nn.MSELoss()(pred, (x_1 - x_0))
            
            acc += loss.item()
            denom += pred.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # wandb.log({"loss": acc/denom})

        print(f'epoch {epoch}, loss {acc/denom}')
        torch.save(model.state_dict(), f'celeba.pt')

        randn = torch.randn_like(x_alpha).cuda()
        sampled_imgs = sample(randn, T, model, save = False)

        # images = wandb.Image(
        #     torchvision.utils.make_grid(sampled_imgs), 
        #     caption="generated faces"
        #     )
            
        # wandb.log({"generated faces": images})



def sample(x_0, T, model, save = False):
    if save:
        if not os.path.exists("temp_dir/"):
            os.mkdir("temp_dir")

    with torch.no_grad():
        schedule = (torch.arange(0, T+1)/T).cuda()

        curr = x_0
        for t in tqdm.tqdm(range(0, T)):
            schedule_future_brod = schedule[t+1].repeat((x_0.shape[0]))
            schedule_curr_brod = schedule[t].repeat((x_0.shape[0]))

            curr = curr + (schedule_future_brod[..., None, None, None] - schedule_curr_brod[...,
                           None, None, None]) * model(curr, schedule_curr_brod)['sample']
            if save:
                torchvision.utils.save_image(curr, f'temp_dir/{t}.png')
    return curr


if __name__ == "__main__":
    train()

    model = get_model().cuda()
    model.load_state_dict(torch.load("celeba.pt"))
    # model.eval()

    # x_0 = torch.randn(1, 3, 256, 256).cuda()
    # ret = sample(x_0, T, model)

    # images = []
    # for img in sorted(os.listdir("temp_dir/"), key = lambda x: int(x.split(".")[0])):
    #     images.append(imageio.imread(f'temp_dir/{img}'))

    # f = 'rgb_video.mp4'
    # imageio.mimwrite(f, images, fps=30, quality=7)

