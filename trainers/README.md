# Usage

Refer to scripts to get all the converted initializations to match HF. I was too
lazy to look into how the initialization was done, so we just load the weights -
but I'll go back to it when I need to.

```
python3 trainers/lightning_trainer.py  fit --data.dataset_path=/home/bocchi/datasets --data.model_path=/home/bocchi/models/google-bert/bert-large-uncased-whole-word-masking/ --model.path_to_init_weights=/home/bocchi/bert/models/custom_initialization/google-bert/bert-large-uncased-whole-word-masking --trainer.callbacks+=LearningRateMonitor --trainer.callbacks.logging_interval=step --seed_everything 42 --trainer.logger=WandbLogger --trainer.logger.project=pretrain-bert --model.learning_rate=1e-04 --model.num_warmup_steps=8000 --model.num_training_steps=50000 --data.batch_size=64 --trainer.accumulate_grad_batches=1 --trainer.callbacks+=ModelCheckpoint --trainer.callbacks.every_n_train_steps=1000 --trainer.callbacks.save_top_k=-1
```

# Notes

When you are first starting out, you're basically just setting hyperparameters
randomly. This is why "hyperparameter search" exists in the first place. You
have to do a few runs and feel out the model, dataset, and gradients. 

1. implementation is correct (compare to a reference implementation from pre-trained models)
2. model is learning (overfitting on a small dataset)
3. optimize training iteration speed
4. hyperparameter search
5. checkpointing + experiments

The awkward thing is that you want to optimizing training speed to make it
faster to run hyperparameter searches... but it actually requires deep knowledge
about torch/jax/etc. + hardware you have available.

It's easy to fall into the trap of getting deep into CUDA before ever running an
experiment. You need to feel things out in parallel to learning if you want to
end up satisfied with your rate of progress.