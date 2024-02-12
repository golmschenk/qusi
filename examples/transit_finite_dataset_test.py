import torch
from torch.nn import BCELoss
from torchmetrics.classification import BinaryAccuracy

from qusi.finite_test_session import finite_datasets_test_session, get_device
from qusi.hadryss_model import Hadryss

from transit_dataset import get_transit_finite_test_dataset

def main():
    test_light_curve_dataset = get_transit_finite_test_dataset()
    model = Hadryss.new()
    device = get_device()
    model.load_state_dict(torch.load('sessions/<wandb_run_name>_latest_model.pt', map_location=device))
    metric_functions = [BinaryAccuracy(), BCELoss()]
    results = finite_datasets_test_session(test_datasets=[test_light_curve_dataset], model=model,
                                           metric_functions=metric_functions, batch_size=100, device=device)
    print(f'Binary accuracy: {results[0][0]}')
    print(f'Binary cross entropy: {results[0][1]}')


if __name__ == '__main__':
    main()
