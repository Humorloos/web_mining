from transformer.emoBert import EmoBERT
import pytorch_lightning as pl

if __name__ == '__main__':
    model = EmoBERT()

    # train model
    trainer = pl.Trainer(
        accelerator="cpu"
    )
    trainer.fit(model)
