import torch
import pandas as pd

from moa_dataset import MoaSurveyMicrolensingAndNonMicrolensingDatabase
from functools import partial
from qusi.finite_standard_light_curve_dataset import FiniteStandardLightCurveDataset
from qusi.light_curve_dataset import default_light_curve_observation_post_injection_transform, \
    default_light_curve_post_injection_transform
from qusi.hadryss_model import Hadryss
from qusi.infer_session import get_device, infer_session


def read_wandb_name_csv(test_split_):
    df = pd.read_csv('inferences/wandb.csv')
    df_temp = df[df["Tags"] == f"550k M vs NM Split {test_split_}"]
    wandb_name = df_temp['Name'].values[0]
    return wandb_name.strip()


def main(test_split_, type_, wandb_name_):
    print('Test split #: ', test_split_)
    print('Type: ', type_)
    print('WANDB name: ', wandb_name_)
    database = MoaSurveyMicrolensingAndNonMicrolensingDatabase(test_split=test_split_)
    infer_light_curve_collection = database.get_microlensing_infer_collection()
    test_light_curve_dataset = FiniteStandardLightCurveDataset.new(
        light_curve_collections=[infer_light_curve_collection])
        # post_injection_transform=partial(
        #     default_light_curve_post_injection_transform, length=18_000))
    # test_light_curve_dataset.post_injection_transform = partial(default_light_curve_observation_post_injection_transform,
    #                                                             length=2500)
    # Change later for input_length = 18_000
    model = Hadryss.new(input_length=2500)
    device = get_device()
    model.load_state_dict(torch.load(f'sessions/{wandb_name_}_latest_model.pt', map_location=device))
    confidences = infer_session(infer_datasets=[test_light_curve_dataset], model=model,
                                batch_size=100, device=device)[0]
    paths = list(database.all_inference.get_paths())
    paths_with_confidences = zip(paths, confidences)
    sorted_paths_with_confidences = sorted(
        paths_with_confidences, key=lambda path_with_confidence: path_with_confidence[1], reverse=True)
    print(sorted_paths_with_confidences)
    df = pd.DataFrame(sorted_paths_with_confidences, columns=['Path', 'Score'])
    df['Path'] = df['Path'].astype(str)
    lightcurves_names = df['Path'].str.split('/').str[-1].str.split('.').str[0].str.split('_').str[-1]
    # .str.split('.')[0].str.split('_')[-1]
    df['lightcurve_name'] = lightcurves_names
    df.to_csv(f'inferences/results_{type_}_{test_split_}.csv')

    print()


if __name__ == '__main__':
    import sys
    import time
    start_time = time.time()
    # total arguments
    n = len(sys.argv)
    print("Total arguments passed:", n)
    # Arguments passed
    python_script_name = sys.argv[0]
    split_number = int(sys.argv[1])
    wandb_name = str(read_wandb_name_csv(split_number))

    main(test_split_=split_number, type_='550k', wandb_name_=wandb_name)

    # main(test_split_=int(0), type_='550k', eventname_='gs66-ponyta')
    # main(test_split_=int(0), type_='550k', eventname_='graceful-serenity-51')
    end_time = time.time()
    print('Time taken: ', end_time - start_time)
    # gs66-fugu-550k
    # main(test_split_=int(0), type_='550k', eventname_='confused-resonance-22')

    # The below work on my computer. above does not work on fugu
    # main(test_split_=int(0), type_='550k', eventname_='gs66-ponyta')
