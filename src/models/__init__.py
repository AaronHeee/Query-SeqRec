from .sasrec_model import SASRecModel

MODELS = {
    "sasrec": SASRecModel,
}

def model_factory(args):
    model = MODELS[args.model_code]
    return model(args)
