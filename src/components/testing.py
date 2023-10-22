import torch

from src.components.utils.settings import Settings
from src.components.model.ddpm import DDPM
from src.components.model.unet import UNet
from src.components.visualization.display_images import show_images


SETTINGS=Settings()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SETTINGS.logger.info(f'Using device: {device}\t' + (f'{torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'cpu'))

# Model
best_model=DDPM(network=UNet(**SETTINGS.unet['DEFAULT']), device=device, **SETTINGS.ddpm['DEFAULT'])
        
# Loading the trained model
best_model.load_state_dict(torch.load('src/assets/ddpm_model.pt', map_location=device))
best_model.eval()
print("Model loaded")

print("Generating new images")
generated = best_model.generate_new_images(
        best_model,
        n_samples=100,
        device=device,
        gif_name="test.gif"
    )
show_images(generated, "Final result")