from .interaction import ItemQueryDataset

DATASETS = {
    ItemQueryDataset.code(): ItemQueryDataset,
}

def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
