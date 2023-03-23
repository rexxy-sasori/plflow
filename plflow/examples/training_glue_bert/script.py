import argparse

import pytorch_lightning as pl

import plflow.training
from plflow import data as datazoo
import plflow.config.parsers as configparsers
from plflow.config.usr_config import get_usr_config
from plflow.models.transformers import TransformerSequenceClassification
from plflow.training.wrappers import GlueSequenceClassificationWrapper

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--usr-config', type=str, required=True)
    command_args = parser.parse_args()
    usr_config = get_usr_config(command_args.usr_config)

    pl.seed_everything(usr_config.seed)

    data_module = configparsers.parse_datamodule(usr_config, datazoo)
    data_module.setup("fit")

    model = TransformerSequenceClassification(
        usr_config.module.init_args.model_name_or_path,
        data_module.num_labels, data_module.task_name
    )

    logger = configparsers.parse_logging(
        usr_config=usr_config,
        use_time_code=usr_config.trainer.init_args.use_time_code,
        name='glue_bert'
    )

    strategy = configparsers.parse_strategy(usr_config.trainer.init_args.training_strategy)
    callbacks = configparsers.parse_callbacks(
        logger, usr_config, pl.callbacks,
        usr_config.trainer.init_args.persist_ckpt
    )

    plmodule = GlueSequenceClassificationWrapper(
        model=model,
        usr_config=usr_config,
        task_name=data_module.task_name,
        num_labels=data_module.num_labels,
        eval_splits=data_module.eval_splits,
        save_usr_config=True
    )

    plflow.training.execute(
        plmodule=plmodule,
        datamodule=data_module,
        num_epochs=usr_config.trainer.init_args.num_epochs,
        ckpt_path=usr_config.trainer.init_args.ckpt_path,
        logger=logger,
        strategy=strategy,
        callbacks=callbacks,
    )
