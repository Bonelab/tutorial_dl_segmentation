from __future__ import annotations

# external imports
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from blpytorchlightning.dataset_components.file_loaders.AIMLoader import AIMLoader
from blpytorchlightning.dataset_components.transformers.Rescaler import Rescaler
from blpytorchlightning.dataset_components.samplers.SliceSampler import SliceSampler
from blpytorchlightning.dataset_components.samplers.ComposedSampler import ComposedSampler
from blpytorchlightning.dataset_components.samplers.ForegroundPatchSampler import ForegroundPatchSampler
from blpytorchlightning.dataset_components.datasets.ComposedDataset import ComposedDataset

import numpy as np
from scipy.ndimage import zoom
import pickle
from tqdm import tqdm
import os


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Load some AIM images and masks, extract slices, downsample, and save in a new directory.",
        formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "data_dir", type=str, metavar="DIR",
        help="directory to pull data from"
    )

    parser.add_argument(
        "output_dir", type=str, metavar="DIR",
        help="directory to save prepped data to"
    )

    return parser


def main(args: Namespace):

    intensity_bounds = -400, 1400

    dataset = ComposedDataset(
        AIMLoader(args.data_dir, '*_*_??.AIM'),
        ComposedSampler([
            SliceSampler(dims=[2]),
            ForegroundPatchSampler(patch_width=128, foreground_channel=0)
        ]),
        Rescaler(intensity_bounds)
    )

    dataset.pickle_dataset(args.output_dir, list(range(100)), 1)


if __name__ == "__main__":
    main(create_parser().parse_args())
