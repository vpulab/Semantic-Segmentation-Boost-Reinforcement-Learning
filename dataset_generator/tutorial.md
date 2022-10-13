# Dataset Generation

This module is used to generate a semantic segmentation dataset with synthetic images that resemble real frames from the Super Mario videogame. Tested with python 3.9

## How to run

Create your python environment and do:

    pip install -r requirements.txt

First, run  It will populate the Sprites folder.

    python extract_subimages.py

Now, run:

    python generate_frames.py -s NUMBER_OF_SAMPLES_TO_GENERATE

If you are running on linux, you can use multiple cores to make the generation faster. Instead, run the command:

    python generate_frames.py -s NUMBER_OF_SAMPLES_TO_GENERATE -c NUMBER_OF_CORES

Dataset will automatically generate in a folder called dataset.