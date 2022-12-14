from .sasrec import SASRecSampleTrainer, SASRecAllTrainer

TRAINERS = {
    SASRecSampleTrainer.code(): SASRecSampleTrainer,
    SASRecAllTrainer.code(): SASRecAllTrainer,
}


def trainer_factory(args, model, train_loader, val_loader, test_loader, ckpt_root, user2seq, embed_only=False):
    trainer = TRAINERS[args.trainer_code]
    return trainer(args, model, train_loader, val_loader, test_loader, ckpt_root, user2seq, embed_only)