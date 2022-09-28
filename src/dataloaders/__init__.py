from src.datasets import dataset_factory

# seq baseline
from .sasrec import SASRecDataloader
# ours
from .aug_sasrec import AUGSASRecDataloader


DATALOADERS = {
    SASRecDataloader.code(): SASRecDataloader,
    AUGSASRecDataloader.code(): AUGSASRecDataloader,
}


def dataloader_factory(args, dataset):
    dataloader = DATALOADERS[args.dataloader_code]
    dataloader = dataloader(args, dataset)
    train, val, test = dataloader.get_pytorch_dataloaders()
    return train, val, test, dataset
