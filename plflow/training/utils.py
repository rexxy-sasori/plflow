import pytorch_lightning as pl
import torch


def run_glue_benchmark(
        pl_module: pl.LightningModule,
        outputs: list,
        log_mnli_key_format: str,
        log_mnli_loss_format: str,
        log_loss_key: str,
):
    if pl_module.task_name == "mnli":
        for i, output in enumerate(outputs):
            split = pl_module.eval_splits[i].split("_")[-1]
            preds = torch.cat([x["preds"] for x in output]).detach().cpu().numpy()
            labels = torch.cat([x["labels"] for x in output]).detach().cpu().numpy()
            loss = torch.stack([x["loss"] for x in output]).mean()
            pl_module.log(log_mnli_loss_format.format(split), loss, prog_bar=True)
            metric_results = pl_module.metric.compute(predictions=preds, references=labels)
            split_metrics = {log_mnli_key_format.format(k, split): v for k, v in metric_results.items()}
            pl_module.log_dict(split_metrics, prog_bar=True)
        return

    preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
    labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
    loss = torch.stack([x["loss"] for x in outputs]).mean()
    pl_module.log(log_loss_key, loss, prog_bar=True)
    pl_module.log_dict(pl_module.metric.compute(predictions=preds, references=labels), prog_bar=True)
