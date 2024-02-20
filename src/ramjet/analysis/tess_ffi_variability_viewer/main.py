import logging
from pathlib import Path

import pandas as pd
from astropy import units
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from ramjet.photometric_database.tess_ffi_light_curve import TessFfiLightCurve

logger = logging.getLogger(__name__)


def create_pdf_of_light_curve_variability(light_curve_path: Path, output_pdf_path: Path) -> None:
    light_curve = TessFfiLightCurve.from_path(light_curve_path)
    (
        fold_period,
        fold_epoch,
        time_bin_size,
        minimum_bin_phase,
        maximum_bin_phase,
        inlier_lightkurve_light_curve,
        periodogram,
        folded_lightkurve_light_curve,
    ) = light_curve.get_variability_phase_folding_parameters_and_folding_lightkurve_light_curves()
    variability_centroid_and_frames = (
        light_curve.estimate_photometric_variability_centroid_and_frames_from_ffi_based_on_folding_parameters(
            fold_epoch, fold_period, maximum_bin_phase, minimum_bin_phase, time_bin_size
        )
    )
    centroid_sky_coord = variability_centroid_and_frames[0]
    target_pixel_file = variability_centroid_and_frames[1]
    difference_target_pixel_frame = variability_centroid_and_frames[2]
    median_maximum_target_pixel_frame = variability_centroid_and_frames[3]
    median_minimum_target_pixel_frame = variability_centroid_and_frames[4]
    with PdfPages(output_pdf_path) as pdf_file:
        inlier_lightkurve_light_curve_figure, inlier_lightkurve_light_curve_axes = plt.subplots()
        inlier_lightkurve_light_curve.scatter(ax=inlier_lightkurve_light_curve_axes)
        pdf_file.savefig(inlier_lightkurve_light_curve_figure)
        periodogram_figure, periodogram_axes = plt.subplots()
        periodogram.plot(ax=periodogram_axes)
        pdf_file.savefig(periodogram_figure)
        folded_lightkurve_light_curve_figure, folded_lightkurve_light_curve_axes = plt.subplots()
        folded_lightkurve_light_curve.scatter(ax=folded_lightkurve_light_curve_axes)
        pdf_file.savefig(folded_lightkurve_light_curve_figure)
        median_minimum_target_pixel_frame_figure, median_minimum_target_pixel_frame_axes = plt.subplots()
        median_minimum_target_pixel_frame.plot(ax=median_minimum_target_pixel_frame_axes)
        pdf_file.savefig(median_minimum_target_pixel_frame_figure)
        median_maximum_target_pixel_frame_figure, median_maximum_target_pixel_frame_axes = plt.subplots()
        median_maximum_target_pixel_frame.plot(ax=median_maximum_target_pixel_frame_axes)
        pdf_file.savefig(median_maximum_target_pixel_frame_figure)
        difference_target_pixel_frame_figure, difference_target_pixel_frame_axes = plt.subplots()
        difference_target_pixel_frame.plot(ax=difference_target_pixel_frame_axes)
        centroid_pixel_position = target_pixel_file.wcs.world_to_pixel(centroid_sky_coord)
        difference_target_pixel_frame_axes.plot(
            target_pixel_file.column + centroid_pixel_position[0],
            target_pixel_file.row + centroid_pixel_position[1],
            "co",
        )
        difference_target_pixel_frame_axes.plot(
            target_pixel_file.column + centroid_pixel_position[0],
            target_pixel_file.row + centroid_pixel_position[1],
            "mo",
        )
        target_sky_coord = SkyCoord(ra=target_pixel_file.ra, dec=target_pixel_file.dec, unit=units.deg)
        target_pixel_position = target_pixel_file.wcs.world_to_pixel(target_sky_coord)
        difference_target_pixel_frame_axes.plot(
            target_pixel_position[0] + target_pixel_file.column, target_pixel_position[1] + target_pixel_file.row, "ro"
        )
        pdf_file.savefig(difference_target_pixel_frame_figure)
        plt.close("all")  # TODO: This is a hack. Should really use proper figure contexts or something.


def create_pdf_of_variability_of_filtered_light_curves_csv(csv_path: Path, output_directory_path: Path):
    filtered_data_frame = pd.read_csv(csv_path)
    # filtered_data_frame = filtered_data_frame.tail(10)
    output_directory_path.mkdir(exist_ok=True, parents=True)
    skipped = []
    for index, row in filtered_data_frame.iterrows():
        light_curve_path = Path(row["light_curve_path"])
        tic_id, sector = row["tic_id"], row["sector"]
        pdf_path = output_directory_path.joinpath(f"{index}_tic_id_{tic_id}_sector_{sector}")
        logger.info("=" * 50)
        logger.info(pdf_path)
        create_pdf_of_light_curve_variability(light_curve_path, pdf_path)
        logger.info("=" * 50)
    logger.info(f"Skipped: {len(skipped)}")
    logger.info(f"{skipped}")


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    filtered_csv_path_ = Path(
        "/att/gpfsfs/briskfs01/ppl/golmsche/generalized-photometric-neural-network-experiments/logs/"
        "FfiHades_corrected_non_rrl_label_no_bn_2022_02_03_23_52_12/filtered_infer_results_2022-02-06-13-21-41.csv"
    )
    output_directory_path_ = filtered_csv_path_.parent.joinpath("variability_pdfs")
    create_pdf_of_variability_of_filtered_light_curves_csv(filtered_csv_path_, output_directory_path_)
