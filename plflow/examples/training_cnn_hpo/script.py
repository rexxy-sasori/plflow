import argparse

import optuna
import pytorch_lightning as pl
from optuna.integration import PyTorchLightningPruningCallback

import plflow.config.parsers as config_parsers
import plflow.data
import plflow.models
import plflow.training
from plflow.config.usr_config import get_usr_config
from plflow.training.wrappers import ImageClassificationWrapper


def pipeline(trial: optuna.trial.Trial):
    usr_config = get_usr_config(path)

    pl.seed_everything(usr_config.seed)

    usr_config.optimizer.label_smoothing = trial.suggest_categorical('label_smoothing', [0.01, 0.05, 0.1, 0.2])
    usr_config.optimizer.reg_bn = trial.suggest_categorical('reg_bn', [True, False])
    usr_config.optimizer.init_args.lr = trial.suggest_categorical('lr', [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1])
    usr_config.optimizer.init_args.weight_decay = trial.suggest_categorical('weight_decay',
                                                                            [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5,
                                                                             8e-5, 9e-5, 1e-4])
    usr_config.data.init_args.batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])

    model = config_parsers.parse_model(usr_config, plflow.models)
    datamodule = config_parsers.parse_datamodule(usr_config, plflow.data)

    logger = config_parsers.parse_logging(
        usr_config=usr_config,
        use_time_code=usr_config.trainer.init_args.use_time_code,
        name='img_classification'
    )

    strategy = config_parsers.parse_strategy(usr_config.trainer.init_args.training_strategy)
    callbacks = config_parsers.parse_callbacks(
        logger, usr_config, pl.callbacks,
        usr_config.trainer.init_args.persist_ckpt
    )

    callbacks += [PyTorchLightningPruningCallback(trial, monitor="val_acc")]

    plmodule = ImageClassificationWrapper(
        model=model, usr_config=usr_config, save_usr_config=True
    )

    trainer = plflow.training.execute(
        plmodule=plmodule,
        datamodule=datamodule,
        num_epochs=usr_config.trainer.init_args.num_epochs,
        ckpt_path=usr_config.trainer.init_args.ckpt_path,
        logger=logger,
        strategy=strategy,
        callbacks=callbacks,
    )

    return trainer.callback_metrics['test_acc'].item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--usr-config', type=str, required=True)
    command_args = parser.parse_args()
    path = command_args.usr_config

    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(pipeline, n_trials=200, show_progress_bar=True, gc_after_trial=True)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
