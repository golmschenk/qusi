import os.path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bokeh.palettes import Category20

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import roc_curve, auc

from stela_investigation.quality_of_results.InferredNeuralNetwork_cls import InferredNeuralNetwork


class AnalysisInferredNN(InferredNeuralNetwork):
    """
    A class to analyse the results of the inference
    """
    def __init__(self, inference_name):
        super().__init__(inference_name)
        # If the file 'infer_results_with_tag.csv' doesn't exist, create:
        if not os.path.exists(self.inference_results_with_matching_tags_path):
            print('Matching the labels from Taka&Yuki with the inference output of our NN...')
            self.complete_label_and_raw_inference_matcher()

        self.inference_with_matching_tags_df = self.inference_with_matching_tags_dataframer()
        self.threshold_value = None
        print('Inference: ', self.inference_folder_name)

    def inference_distribution_plotter(self, show_plot=False):
        # Separating originally labeled microlensing ('c', 'cf', 'cp', 'cw', 'cs', 'cb') from not microlensing
        #         ('v', 'n', 'nr', 'm', 'j', moa_data_interface.no_tag_string)
        labeled_microlensing_df = self.inference_with_matching_tags_df[
            self.inference_with_matching_tags_df['is_microlensing']]
        labeled_not_microlensing_df = self.inference_with_matching_tags_df[
            np.invert(self.inference_with_matching_tags_df['is_microlensing'])]
        labeled_microlensing_organized = labeled_microlensing_df.sort_values(by=['confidence'])
        labeled_not_microlensing_organized = labeled_not_microlensing_df.sort_values(by=['confidence'])

        # calculate CDF values
        labeled_microlensing_cdf = 1. * np.arange(len(labeled_microlensing_organized['confidence'])) / \
                                   (len(labeled_microlensing_organized['confidence']) - 1)
        labeled_not_microlensing_cdf = 1. * np.arange(len(labeled_not_microlensing_organized['confidence'])) / \
                                       (len(labeled_not_microlensing_organized['confidence']) - 1)
        print()

        # Adding 0 and 1 to start and ends
        confidence_labeled_microlensing = labeled_microlensing_organized['confidence'].values
        confidence_labeled_not_microlensing = labeled_not_microlensing_organized['confidence'].values

        labeled_microlensing_cdf = np.insert(labeled_microlensing_cdf, 0, 0)
        labeled_microlensing_cdf = np.insert(labeled_microlensing_cdf, len(labeled_microlensing_cdf), 1)
        confidence_labeled_microlensing = np.insert(confidence_labeled_microlensing, 0, 0)
        confidence_labeled_microlensing = np.insert(confidence_labeled_microlensing, len(confidence_labeled_microlensing), 1)

        labeled_not_microlensing_cdf = np.insert(labeled_not_microlensing_cdf, 0, 0)
        labeled_not_microlensing_cdf = np.insert(labeled_not_microlensing_cdf, len(labeled_not_microlensing_cdf), 1)
        confidence_labeled_not_microlensing = np.insert(confidence_labeled_not_microlensing, 0, 0)
        confidence_labeled_not_microlensing = np.insert(confidence_labeled_not_microlensing, len(confidence_labeled_not_microlensing), 1)


        fig, ax = plt.subplots()
        ax.step(confidence_labeled_microlensing, labeled_microlensing_cdf,
                label='Microlensing')
        ax.step(confidence_labeled_not_microlensing, labeled_not_microlensing_cdf,
                label='Not Microlensing', color='orange')
        ax.set(xlabel='Neural Network Confidence', ylabel='Probability',
               title=f'Cumulative Distribution {self.inference_folder_name}')
        plt.legend()
        plt.savefig(f'stela_investigation/Crossvalidation_inference_plots/'
                    f'confidence_distribution_{self.inference_folder_name}.png', dpi=300)
        if show_plot:
            plt.show()
        plt.close()

    def inference_distribution_per_tag_plotter(self, show_plot=False):
        fig, ax = plt.subplots()
        tags = ['v', 'n', 'nr', 'm', 'j', 'no_tag', 'c', 'cf', 'cp', 'cw', 'cs', 'cb']
        for tag, color_index in zip(tags, np.arange(0, len(tags))):
            the_tag_only_df = self.inference_with_matching_tags_df[self.inference_with_matching_tags_df['tag'] == tag].sort_values(by=['confidence'])
            # calculate CDF values
            the_tag_only_cdf = 1. * np.arange(len(the_tag_only_df['confidence'])) \
                               / (len(the_tag_only_df['confidence']) - 1)

            # Adding 0 and 1 to start and ends
            confidence_tag_only = the_tag_only_df['confidence'].values
            the_tag_only_cdf = np.insert(the_tag_only_cdf, 0, 0)
            the_tag_only_cdf = np.insert(the_tag_only_cdf, len(the_tag_only_cdf), 1)
            confidence_tag_only = np.insert(confidence_tag_only, 0, 0)
            confidence_tag_only = np.insert(confidence_tag_only, len(confidence_tag_only), 1)

            ax.step(confidence_tag_only, the_tag_only_cdf,
                    label=f'{tag} #{len(the_tag_only_cdf)}', color=Category20[20][color_index])

        ax.set(xlabel='Neural Network Confidence', ylabel='Probability',
               title=f'Cumulative Distribution {self.inference_folder_name}')
        plt.legend()
        plt.savefig(f'stela_investigation/Crossvalidation_inference_plots/'
                    f'confidence_distribution_per_tag_{self.inference_folder_name}.png', dpi=300)
        if show_plot:
            plt.show()
        plt.close()

    def confusion_matrix_plotter(self, should_normalize_=None,
                                 labels_=['Not \n Microlensing', 'Microlensing'],
                                 show_plot=False):
        disp = ConfusionMatrixDisplay.from_predictions(self.inference_with_matching_tags_df['is_microlensing'],
                                                       self.inference_with_matching_tags_df['prediction'],
                                                       display_labels=labels_,
                                                       cmap=plt.cm.Blues,
                                                       normalize=should_normalize_)

        plt.title(f'Confusion Matrix - Threshold: {self.threshold_value} -{self.inference_folder_name}')
        plt.tight_layout()
        if should_normalize_ == 'true':
            plt.savefig(f'stela_investigation/Crossvalidation_inference_plots/normalized_confusion_matrix_'
                        f'{self.threshold_value}_{self.inference_folder_name}.png', dpi=300)
        else:
            plt.savefig(f'stela_investigation/Crossvalidation_inference_plots/confusion_matrix_'
                        f'{self.threshold_value}_{self.inference_folder_name}.png', dpi=300)
        if show_plot:
            plt.show()
        plt.close()

    def ROC_plotter(self, show_plot=False):
        """
        From this example https://scikit-learn.org/stable/auto_examples/
        model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
        :param dataframe_:
        :return:
        """
        # Compute ROC curve and ROC area for each class
        false_positive_rate, true_positive_rate, _ = roc_curve(self.inference_with_matching_tags_df['Microlensing_1or0'],
                                                               self.inference_with_matching_tags_df['confidence'])
        roc_auc = auc(false_positive_rate, true_positive_rate)

        plt.figure()
        plt.plot(false_positive_rate, true_positive_rate,
                 color="maroon", lw=3,
                 label="ROC curve (area = %0.5f)" % roc_auc)
        # diagonal
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Diagonal Line TPR = FPR")
        # plt.xlim([-0.05, 1.0])
        # plt.ylim([-0.05, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Receiver Operating Characteristic - Microlensing Classification {self.inference_folder_name}")
        plt.legend(loc="lower right")
        plt.savefig(f'stela_investigation/Crossvalidation_inference_plots/ROC_{self.inference_folder_name}.png', dpi=300)
        if show_plot:
            plt.show()
        plt.close()

    def threshold_inference_prediction_setter(self, threshold_):
        """
        This sets the "prediction" columns based on the
        threshold
        :param threshold_:
        :return:
        """
        self.threshold_value = threshold_
        self.inference_with_matching_tags_df['prediction'] = np.where(self.inference_with_matching_tags_df['confidence'] >= self.threshold_value, True, False)
        print()

    def performance_calculator(self, threshold_):
        self.threshold_inference_prediction_setter(threshold_)
        print('Calculating performance using threshold: ', self.threshold_value)
        actual_label_ = self.inference_with_matching_tags_df['is_microlensing']
        predicted_label_ = self.inference_with_matching_tags_df['prediction']
        true_negatives, false_positives, false_negatives, true_positives = confusion_matrix(actual_label_,
                                                                                            predicted_label_).ravel()
        return true_positives, false_positives, true_negatives, false_negatives


def cross_validation_concatenater(log_names, new_path):

    for log_name in log_names:
        inference_object = AnalysisInferredNN(log_name)
        new_df = inference_object.inference_with_matching_tags_dataframer()
        if log_name.split('_')[3] == '0':
            previous_df = new_df
        else:
            previous_df = pd.concat([previous_df, new_df], axis=0, ignore_index=True)

    previous_df.to_csv(f'{new_path}infer_results_with_tag.csv')
    # pd.concat([df1, df2, df3, df4], axis=1, ignore_index=True)


if __name__ == '__main__':
    test0 = AnalysisInferredNN('Hades_crossvalidation')
    test0.inference_distribution_plotter()

    print()
