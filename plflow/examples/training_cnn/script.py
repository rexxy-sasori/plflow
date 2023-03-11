import argparse

import pytorch_lightning as pl

import plflow.config_parsing.parsers as config_parsers
import plflow.data
import plflow.models
import plflow.training
from plflow.config_parsing.usr_config import get_usr_config
from plflow.training.wrappers import ImageClassificationWrapper

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--usr-config', type=str, required=True)
    command_args = parser.parse_args()
    usr_config = get_usr_config(command_args.usr_config)

    pl.seed_everything(usr_config.seed)

    model = config_parsers.parse_model(usr_config, plflow.models)
    datamodule = config_parsers.parse_datamodule(usr_config, plflow.data)

    logger = config_parsers.parse_logging(
        usr_config=usr_config,
        use_time_code=usr_config.trainer.use_time_code,
        name='img_classification'
    )

    strategy = config_parsers.parse_strategy(usr_config.trainer.init_args.training_strategy)
    callbacks = config_parsers.parse_callbacks(
        logger, usr_config, pl.callbacks,
        usr_config.trainer.init_args.persist_ckpt
    )

    plmodule = ImageClassificationWrapper(model=model, usr_config=usr_config, save_usr_config=True)

    plflow.training.execute(
        plmodule=plmodule,
        datamodule=datamodule,
        num_epochs=usr_config.trainer.init_args.num_epochs,
        ckpt_path=usr_config.trainer.init_args.ckpt_path,
        logger=logger,
        strategy=strategy,
        callbacks=callbacks,
    )
