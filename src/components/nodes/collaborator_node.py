from torch.utils.data import DataLoader, Dataset

from torchvision.transforms import Compose, ToTensor, Lambda
from torchvision.datasets.mnist import MNIST, FashionMNIST

from src.components.utils.settings import Settings
from src.components.nodes.base_node import BaseNode

SETTINGS=Settings()

class Collaborator(BaseNode):
    def __init__(self, device, dataset_function=MNIST):
        id='COLLABORATOR_1'
        node_type='Collaborator'
        super().__init__(id, node_type, device=device)
        # Data Handling
        self.transform = Compose([ToTensor(),Lambda(lambda x: (x - 0.5) * 2)])
        self.ds = dataset_function("./data", download=True, train=True, transform=self.transform)
        
    def set_dl(self, batch_size) -> DataLoader:
        # Data Loader
        self.dl=DataLoader(self.ds, batch_size, shuffle=True)