from qusi.hadryss_model import Hadryss
from qusi.train_hyperparameter_configuration import TrainHyperparameterConfiguration
from qusi.train_logging_configuration import TrainLoggingConfiguration
from qusi.train_session import train_session

from torchmetrics.classification import (BinaryAccuracy, BinaryAUROC, BinaryF1Score, BinarySpecificity)

from moa_dataset_2M import MoaSurveyMicrolensingAndNonMicrolensingDatabase2M
from wrapped_metrics import WrappedBinaryPrecision, WrappedBinaryRecall

from tqdm import tqdm

def main(test_split):
    # WAND
    logging_configuration = TrainLoggingConfiguration.new(wandb_project='qusi_moa_2M', wandb_entity='ramjet')

    # Database
    database = MoaSurveyMicrolensingAndNonMicrolensingDatabase2M(test_split=test_split)
    train_light_curve_dataset = database.get_microlensing_train_dataset()
    validation_light_curve_dataset = database.get_microlensing_validation_dataset()

    # model and config
    model = Hadryss.new(input_length=18_000)
    train_hyperparameter_configuration = TrainHyperparameterConfiguration.new(
        batch_size=100, cycles=500, train_steps_per_cycle=500, validation_steps_per_cycle=50)

    metric_functions = [BinaryAccuracy(), BinaryAUROC(), BinaryF1Score(), BinarySpecificity(),
                        WrappedBinaryPrecision(), WrappedBinaryRecall()]

    # Train!
    train_session(train_datasets=[train_light_curve_dataset], validation_datasets=[validation_light_curve_dataset],
                  model=model, hyperparameter_configuration=train_hyperparameter_configuration,
                  logging_configuration=logging_configuration, metric_functions=metric_functions)

if __name__ == '__main__':
    import sys
    # total arguments
    n = len(sys.argv)
    print("Total arguments passed:", n)
    # Arguments passed
    print("\nName of Python script:", sys.argv[0])
    print("\nSplit #:", sys.argv[1])
    # for i in tqdm([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
    #     print('split is ', i)
    #     main(test_split=i)

    main(test_split=int(sys.argv[1]))
