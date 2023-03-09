from transformers import AutoConfig, AutoModelForSequenceClassification, PretrainedConfig


def TransformerSequenceClassification(
        model_name_or_path: str,
        num_labels,  # data args
        task_name,  # data args
        config_name=None,
        cache_dir=None,
        model_revision='main',
        use_auth_token=False,
        ignore_mismatched_sizes=False,
        label_list=None
):
    config = AutoConfig.from_pretrained(
        config_name if config_name else model_name_or_path,
        num_labels=num_labels,  # args from data
        finetuning_task=task_name,  # args from data
        cache_dir=cache_dir,
        revision=model_revision,
        use_auth_token=True if use_auth_token else None,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
        cache_dir=cache_dir,
        revision=model_revision,
        use_auth_token=True if use_auth_token else None,
        ignore_mismatched_sizes=ignore_mismatched_sizes,
    )

    if label_list is not None:
        # Some models have set the order of the labels to use, so let's make sure we do use it.
        label_to_id = None
        if (
                model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
                and task_name is not None
        ):
            # Some have all caps in their config, some don't.
            label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
            if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
                label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
            else:
                raise ValueError(
                    "Your model seems to have been trained with labels, but they don't match the dataset: ",
                    f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                    "\nIgnoring the model labels as a result.",
                )
        elif task_name is None:
            label_to_id = {v: i for i, v in enumerate(label_list)}

        if label_to_id is not None:
            model.config.label2id = label_to_id
            model.config.id2label = {id: label for label, id in config.label2id.items()}
        elif task_name is not None:  # in this path
            model.config.label2id = {l: i for i, l in enumerate(label_list)}
            model.config.id2label = {id: label for label, id in config.label2id.items()}
    else:
        print('label list is None!')
    return model
