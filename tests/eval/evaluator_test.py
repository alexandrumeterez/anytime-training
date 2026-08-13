import pytest
import torch
from torchmetrics import MeanMetric

from olmo.config import EvaluatorType
from olmo.eval.evaluator import Evaluator


@pytest.mark.parametrize("label", ["all-small-ppl-validation_ema_0.0", "all-small-ppl-validation_ema_100.0"])
def test_lm_perplexity_key_includes_evaluator_label(label):
    metric = MeanMetric()
    metric.update(torch.tensor(2.0))
    evaluator = Evaluator(
        label=label,
        type=EvaluatorType.lm,
        eval_loader=None,  # type: ignore[arg-type]
        eval_metric={"c4_val": metric},
    )

    result = evaluator.compute_metrics()

    assert result[f"eval/c4_val/{label}/CrossEntropyLoss"] == pytest.approx(2.0)
    assert result[f"eval/c4_val/{label}/Perplexity"] == pytest.approx(torch.exp(torch.tensor(2.0)).item())
    assert "eval/c4_val/Perplexity" not in result


def test_non_ema_evaluator_keeps_generic_perplexity_alias():
    metric = MeanMetric()
    metric.update(torch.tensor(2.0))
    evaluator = Evaluator(
        label="all-small-ppl-validation",
        type=EvaluatorType.lm,
        eval_loader=None,  # type: ignore[arg-type]
        eval_metric={"c4_val": metric},
    )

    result = evaluator.compute_metrics()

    assert result["eval/c4_val/Perplexity"] == pytest.approx(torch.exp(torch.tensor(2.0)).item())
