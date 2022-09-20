#Tells the script not to use the GPU:
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
"""Code for inference on the contents of a directory."""
import datetime
from pathlib import Path
# from ramjet.models.cura import Cura
from ramjet.models.hades import Hades
from ramjet.photometric_database.derived.moa_survey_microlensing_and_non_microlening_database import \
    MoaSurveyMicrolensingAndNonMicroleningDatabase
from ramjet.analysis.model_loader import get_latest_log_directory
from ramjet.trial import infer

log_names = ['Hades_test_split_0_2022_06_09_17_18_13',
             'Hades_test_split_1_2022_06_18_15_19_42',
             'Hades_test_split_2_2022_07_08_13_58_50',
             'Hades_test_split_3_2022_07_12_16_02_12',
             'Hades_test_split_4_2022_07_20_14_25_12',
             'Hades_test_split_5_2022_07_25_11_30_34',
             'Hades_test_split_6_2022_07_28_11_50_57',
             'Hades_test_split_7_2022_08_01_14_51_29',
             'Hades_test_split_8_2022_08_04_11_47_20',
             'Hades_test_split_9_2022_08_05_14_04_06']


for log_name in log_names:
    # log_name = get_latest_log_directory(logs_directory='logs')  # Uses the latest model in the log directory.
    # log_name = 'logs/Hades_2022_04_18_17_57_55'  # Specify the path to the model to use.
    test_split_number = int(log_name.split('_')[3])

    print(f'Inference starting for test split #{test_split_number}, with fit model {log_name}')
    saved_log_directory = Path(f'logs/{log_name}')
    datetime_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    print('Setting up dataset...', flush=True)
    database = MoaSurveyMicrolensingAndNonMicroleningDatabase(test_split=test_split_number)
    inference_dataset = database.generate_inference_dataset()

    print('Loading model...', flush=True)
    model = Hades(database.number_of_label_values)
    # model = Cura(database.number_of_label_values)
    model.load_weights(str(saved_log_directory.joinpath('best_validation_model.ckpt'))).expect_partial()
    # model.load_weights(str(saved_log_directory.joinpath('latest_model.ckpt'))).expect_partial()

    print('Inferring...', flush=True)
    infer_results_path = saved_log_directory.joinpath(f'infer_results.csv')
    # infer_results_path = saved_log_directory.joinpath(f'infer results {datetime_string}.csv')
    infer(model, inference_dataset, infer_results_path)
