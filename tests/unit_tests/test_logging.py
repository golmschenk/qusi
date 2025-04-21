from torch.nn import BCELoss
from torchmetrics.classification import BinaryAUROC

from qusi.internal.logging import camel_case_acronyms, get_metric_name


def test_camel_case_acronyms():
    assert camel_case_acronyms('BCEntropy') == 'BcEntropy'
    assert camel_case_acronyms('BinaryAUROC') == 'BinaryAuroc'

def test_get_metric_name():
    assert get_metric_name(BCELoss()) == 'bce'
    assert get_metric_name(BinaryAUROC()) == 'binary_auroc'
