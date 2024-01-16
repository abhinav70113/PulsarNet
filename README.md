# PulsarNet

PulsarNet is the first machine learning based frequency domain acceleration search pipeline to find pulsar binaries.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
<!-- - [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements) -->

## Prerequisites

1. [Conda](https://docs.conda.io/en/latest/miniconda.html) - Used for managing environments and dependencies.

## Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/abhinav70113/PulsarNet.git
    cd PulsarNet
    ```

2. **Set up the Conda environment**:

    Create a Conda environment using the provided `environment.yml` file:

    ```bash
    conda env create -f environment.yml
    ```

    Activate the environment:

    ```bash
    conda activate tf-gpu4
    ```

3. **Download Model Binaries**:

    Navigate to the [releases page](https://github.com/abhinav70113/PulsarNet/releases) and download the necessary model binaries. Place them in the "models" directory.

## Usage

1. Ensure you have the model binaries and other necessary files in place.

2. Modify `model_settings.cfd` if necessary, to suit your needs. Make sure to enter the location of Presto and PulsarNet singularity image.

3. For using the gpu, run the following inside the mounted singularity image.

    ```bash
    python main.py None --check_gpu
    ```
    If running on a cluster:
    ```
    srun --cpus-per-task=1 --gres=gpu:1 singularity exec --nv -H $HOME:/home1 -B /hercules:/hercules/ /u/atya/singularity_images/pulsarnet.sif python main.py None --check_gpu
    ```
    
3. Run the main script:

    ```bash
    python main.py time_series_example.dat
    ```
4. This will output a candidate list in the directory "output".

5. Fold the desired candidate using:

    ```bash
    python fold.py --cand 1 --only_cmd output/time_series_example_PulsarNet.txt 
    ```
    This will output the commands to fold the time series file using presto. Additionally if you have the singularity image of [Presto](https://github.com/scottransom/presto.git) available, specify it's locations in the `config.json` file and remove the `--only_cmd` flag. This will fold the candidates and save them in "output" directory. 

## File Structure

- `config.json` - Configuration file containing model parameters and settings.
- `settings.py` - Contains global constants for the project.
- `file_processing.py` - Handles all file processing tasks.
- `utils.py` - Utility functions used across the project.
- `main.py` - Main driver script for the project.
- `fold.py` - Fold the candidates estimated using main.py.
- `environment.yml` - Conda environment file with required dependencies.

