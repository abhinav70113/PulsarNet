# PulsarNet

PulsarNet is the first machine learning based frequency domain acceleration search pipeline to find pulsar binaries.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Prerequisites

1. [Conda](https://docs.conda.io/en/latest/miniconda.html) - Used for managing environments and dependencies.
2. [Presto](https://github.com/scottransom/presto.git) Singularity Image - Used for folding the candidates after the search is complete

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

    Navigate to the [releases page](https://github.com/abhinav70113/PulsarNet.git/releases) and download the necessary model binaries. Place them in the "models" directory (mention the directory if it's specific).

## Usage

1. Ensure you have the model binaries and other necessary files in place.

2. Modify `config.json` if necessary, to suit your needs. Especially make sure to enter the location of Presto singularity image.

3. Run the main script:

    ```bash
    python main.py time_series_example.dat
    ```
4. This will output a candidate list in the directory "output".

5. Fold the desired candidate using:

    ```bash
    python fold.py --cand 1 output/time_series_example_PulsarNet.txt 
    ```
    This will output a folded profile in the directory "output". Optionally, you can print only the folding command using the `--only_cmd` flag.

## File Structure

- `config.json` - Configuration file containing model parameters and settings.
- `settings.py` - Contains global constants for the project.
- `file_processing.py` - Handles all file processing tasks.
- `utils.py` - Utility functions used across the project.
- `main.py` - Main driver script for the project.
- `fold.py` - Fold the candidates estimated using main.py.
- `environment.yml` - Conda environment file with required dependencies.

