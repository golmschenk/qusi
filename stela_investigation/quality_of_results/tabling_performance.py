import pandas as pd

from stela_investigation.quality_of_results.AnalysisInferredNN_cls import AnalysisInferredNN


def table_performance_saver(inference_object, threshold_collection, type_of_run):
    """
    Save as .csv files the performance
    :return:
    """
    # dataframes
    print(f'You will save the : {inference_object.inference_folder_name}\033[0m ')
    print(f'With the thresholds : \033[1;31;40m {threshold_collection}\033[0m ')
    true_positives_column = []
    false_positives_column = []
    true_negatives_column = []
    false_negatives_column = []
    true_positives_percentage_column = []
    false_positives_percentage_column = []
    true_negatives_percentage_column = []
    false_negatives_percentage_column = []

    for threshold in threshold_collection:
        true_positives, false_positives, true_negatives, false_negatives = inference_object.performance_calculator(threshold)
        true_positives_column.append(true_positives)
        false_positives_column.append(false_positives)
        true_negatives_column.append(true_negatives)
        false_negatives_column.append(false_negatives)
        true_positives_percentage_column.append(true_positives / len(inference_object.inference_with_matching_tags_df['prediction']))
        false_positives_percentage_column.append(false_positives / len(inference_object.inference_with_matching_tags_df['prediction']))
        true_negatives_percentage_column.append(true_negatives / len(inference_object.inference_with_matching_tags_df['prediction']))
        false_negatives_percentage_column.append(false_negatives / len(inference_object.inference_with_matching_tags_df['prediction']))

    temp_dictionary = {'thresholds': threshold_collection,
                       'true_positives_column': true_positives_column,
                       'false_positives_column': false_positives_column,
                       'true_negatives_column': true_negatives_column,
                       'false_negatives_column': false_negatives_column,
                       'true_positives_percentage_column': true_positives_percentage_column,
                       'false_positives_percentage_column': false_positives_percentage_column,
                       'true_negatives_percentage_column': true_negatives_percentage_column,
                       'false_negatives_percentage_column': false_negatives_percentage_column}

    table_of_performance = pd.DataFrame(temp_dictionary)
    table_of_performance.set_index('thresholds', inplace=True)
    table_of_performance.to_csv(f'stela_investigation/{type_of_run}_inference_tables/performance_{inference_object.inference_folder_name}.csv')


if __name__ == '__main__':
    threshold_values = [0.5, 0.6]
    inference_object = AnalysisInferredNN('Hades_test_split_1_2022_06_18_15_19_42')
    table_performance_saver(inference_object, threshold_collection=threshold_values)