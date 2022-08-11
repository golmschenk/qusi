from pathlib import Path
import pandas as pd

from tqdm import tqdm

from ramjet.data_interface.moa_data_interface import MoaDataInterface
from ramjet.photometric_database.derived.moa_survey_light_curve_collection import MoaSurveyLightCurveCollection


class InferredNeuralNetwork:
    """
    A class to control the Inference
    """
    def __init__(self, inference_name):
        self.inference_folder_name = inference_name
        self.inference_folder_path = Path(f'logs/{self.inference_folder_name}')
        self.raw_inference_results_path = self.inference_folder_path.joinpath(f'infer_results.csv')
        self.inference_results_with_matching_tags_path = self.inference_folder_path.joinpath('infer_results_with_tag.csv')

    def complete_label_and_raw_inference_matcher(self):
        """
        Matching the labels from Taka&Yuki with the inference output of our NN
        There might be a faster way, but this is it for now
        :return:
        """
        print('Reading raw inference table...')
        raw_inference_df = pd.read_csv(self.raw_inference_results_path)

        print('MOA interface...')
        moa_data_interface = MoaDataInterface()

        negative = MoaSurveyLightCurveCollection(
            survey_tags=['v', 'n', 'nr', 'm', 'j', moa_data_interface.no_tag_string],
            label=0,
            dataset_splits=list(range(10)))
        positive = MoaSurveyLightCurveCollection(survey_tags=['c', 'cf', 'cp', 'cw', 'cs', 'cb'],
                                                 label=1,
                                                 dataset_splits=list(range(10)))
        print('Collecting path... [This might take a while!]')
        negative.get_paths()
        positive.get_paths()
        ###
        print('Creating dataframe with the tags...')
        full_df = pd.DataFrame(columns=['light_curve_path', 'tag', 'is_microlensing'])
        for tag in tqdm(
                ['v', 'n', 'nr', 'm', 'j', moa_data_interface.no_tag_string, 'c', 'cf', 'cp', 'cw', 'cs', 'cb']):
            if tag == 'c' or tag == 'cf' or tag == 'cp' or tag == 'cw' or tag == 'cs' or tag == 'cb':
                label_data = {'light_curve_path': moa_data_interface.survey_tag_to_path_list_dictionary[tag],
                              'tag': [tag] * len(moa_data_interface.survey_tag_to_path_list_dictionary[tag]),
                              'is_microlensing': True}
            else:
                label_data = {'light_curve_path': moa_data_interface.survey_tag_to_path_list_dictionary[tag],
                              'tag': [tag] * len(moa_data_interface.survey_tag_to_path_list_dictionary[tag]),
                              'is_microlensing': False}
            the_mini_df = pd.DataFrame(label_data)
            full_df = full_df.append(the_mini_df, ignore_index=True)
            print(tag)
        full_df['light_curve_path'] = full_df.light_curve_path.astype(str)

        print('Merging dataframe with the inference scores...')
        merged_df = raw_inference_df.merge(full_df, how='left', on=['light_curve_path'])
        merged_df.drop(columns=['index'], inplace=True)

        print('Saving as a CSV file...')
        merged_df.to_csv(self.inference_results_with_matching_tags_path)

    def inference_with_matching_tags_dataframer(self):
        """
        This read the inference file WITH TAG . This is the inference file from our NN + run the python script
        complete_label_and_raw_inference_matcher()
        :return: dataframe
        """
        inference_df = pd.read_csv(self.inference_results_with_matching_tags_path)

        inference_df[['upper_folder', 'up_folder', 'label_folder', 'full_name_event']] = inference_df[
            'light_curve_path']. \
            str.split('/', expand=True)

        inference_df[['field', 'band', 'chip', 'reduced_name']] = inference_df['full_name_event'].str.split('_',
                                                                                                            expand=True)
        inference_df[['moa_intern_name', 'phot', 'cor', 'feather']] = inference_df['reduced_name'].str.split('.',
                                                                                                             expand=True)
        inference_df.drop(columns=['full_name_event', 'field', 'band', 'chip', 'reduced_name', 'phot', 'cor', 'feather',
                                   'upper_folder', 'up_folder', 'label_folder'],
                          inplace=True)
        inference_df[['field', 'band', 'chip', 'subframe', 'ID']] = inference_df['moa_intern_name'].str.split('-',
                                                                                                              expand=True)

        inference_df['Microlensing_1or0'] = inference_df['is_microlensing'] * 1.0
        return inference_df

if __name__ == '__main__':
    test0 = InferredNeuralNetwork('Hades_test_split_0_2022_06_09_17_18_13')

    print()