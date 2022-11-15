import pandas as pd
from tqdm import tqdm
import matplotlib
# matplotlib.use("TkAgg")

from stela_investigation.quality_of_results.AnalysisInferredNN_cls import AnalysisInferredNN, \
    cross_validation_concatenater
from stela_investigation.quality_of_results.tabling_performance import table_performance_saver


def main_quality_control(inference_object_, threshold_values_, type_of_run_):

    # Plot cumulative distribution
    print('MQC inference path:', inference_object_.inference_folder_path)
    print('Plotting cumulative distribution...')
    inference_object_.inference_distribution_plotter(type_of_run=type_of_run_)
    print('Plotting cumulative distribution per tag...')
    inference_object_.inference_distribution_per_tag_plotter(type_of_run=type_of_run_)

    # Plot confusion Matrix
    print('Plotting Confusion Matrix...')
    for threshold in tqdm(threshold_values_):
        inference_object_.threshold_inference_prediction_setter(threshold)
        inference_object_.confusion_matrix_plotter(type_of_run=type_of_run_)
        # normalized result
        inference_object_.confusion_matrix_plotter(type_of_run=type_of_run_, should_normalize_='true')

    # Plot ROC
    print('Plotting ROC')
    inference_object_.ROC_plotter(type_of_run=type_of_run_)

    # Table of Performance
    print('Creating table of performance')
    table_performance_saver(inference_object_, threshold_collection=threshold_values_, type_of_run=type_of_run_)

    # Plot gbxx of
    # TP
    # FP
    # TN
    # FN

if __name__ == '__main__':
    # # ===========================================
    # # Plotting separately
    # # ===========================================
    threshold_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99]
    # threshold_values = [0.5, 0.6]

    # log_names = ['Hades_test_split_0_2022_06_09_17_18_13',
    #              'Hades_test_split_1_2022_06_18_15_19_42',
    #              'Hades_test_split_2_2022_07_08_13_58_50',
    #              'Hades_test_split_3_2022_07_12_16_02_12',
    #              'Hades_test_split_4_2022_07_20_14_25_12',
    #              'Hades_test_split_5_2022_07_25_11_30_34',
    #              'Hades_test_split_6_2022_07_28_11_50_57',
    #              'Hades_test_split_7_2022_08_01_14_51_29',
    #              'Hades_test_split_8_2022_08_04_11_47_20',
    #              'Hades_test_split_9_2022_08_05_14_04_06']
    # ===========================================
    # TIME
    # ===========================================
    log_names = ['Hades_TIME_test_split_0_2022_10_17_09_50_31',
                 'Hades_TIME_test_split_1_2022_10_12_14_40_41',
                 'Hades_TIME_0and1']
    type_of_run = 'TIME'
    for log_name in tqdm(log_names):
        inference_object = AnalysisInferredNN(log_name)
        main_quality_control(inference_object, threshold_values, type_of_run)

    # # ===========================================
    # # Hard easy
    # # ===========================================
    # log_names = ['Hades_hard_cases_test_split_0_2022_09_21_10_09_07',
    #              'Hades_hard_cases_test_split_1_2022_09_23_15_23_34',
    #              'Hades_hard_cases_0and1']
    # type_of_run = 'Hard_Easy'
    # for log_name in tqdm(log_names):
    #     inference_object = AnalysisInferredNN(log_name)
    #     main_quality_control(inference_object, threshold_values, type_of_run)
    # # ===========================================
    # # Standard
    # # ===========================================
    # log_names = ['Hades_0and1']
    # type_of_run = 'Crossvalidation'
    # for log_name in tqdm(log_names):
    #     inference_object = AnalysisInferredNN(log_name)
    #     main_quality_control(inference_object, threshold_values, type_of_run)

    # ===========================================
    # concatenater
    # ===========================================
    # log_names = ['Hades_test_split_0_2022_06_09_17_18_13',
    #              'Hades_test_split_1_2022_06_18_15_19_42',
    #              'Hades_test_split_2_2022_07_08_13_58_50',
    #              'Hades_test_split_3_2022_07_12_16_02_12',
    #              'Hades_test_split_4_2022_07_20_14_25_12',
    #              'Hades_test_split_5_2022_07_25_11_30_34',
    #              'Hades_test_split_6_2022_07_28_11_50_57',
    #              'Hades_test_split_7_2022_08_01_14_51_29',
    #              'Hades_test_split_8_2022_08_04_11_47_20',
    #              'Hades_test_split_9_2022_08_05_14_04_06']
    # cross_validation_concatenater(log_names, 'logs/Hades_crossvalidation/')

    # threshold_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99]
    # # # threshold_values = [0.5, 0.6]
    # #
    # log_name = 'Hades_crossvalidation'
    # type_of_run = 'Crossvalidation'
    # inference_object = AnalysisInferredNN(log_name)
    # main_quality_control(inference_object, threshold_values, type_of_run)

    print()
