# Spiral_Chronogram

# Chronogram Plotting Package

This package provides a function to plot a chronogram (triangular or spiral) based on a distance matrix and corresponding dates. The main function, `plot_chronogram`, generates these plots with customizable parameters.

## Function: plot_chronogram

### Description

The `plot_chronogram`, `main.py`, function plots a chronogram (triangular or spiral) based on the provided distance matrix and dates.

### Parameters

- `distance_matrix_path` (str): Path to the distance matrix file.
- `dates_file_path` (str): Path to the dates file.
- `threshold` (int, default=1800): Threshold for the spiral.
- `pattern` (str, default='nonoverlap'): Pattern indicating overlap or non-overlap.
- `transform` (bool, default=False): Whether to apply log transformation to angles.
- `aggregate_data` (bool, default=True): Whether to aggregate the data before plotting.
- `gap` (int, default=100): Gap for the spiral.
- `color` (str, default='Spectral'): Color map for the plot.
- `plot_type` (str, default='spiral'): Type of chronogram to plot ('triangular', 'spiral', 'upper_triangle').
- `delta` (str, default='D'): Unit of the dates ("D": days).
- `tick_step` (int, default=100): The increment of highlight tick.

### Example Usage

Run the following command to generate the plot under default settings;

`python main.py --distance_matrix_path <path> --dates_file_path  <path>` 

Command to generate a customized plot;

`python main.py --distance_matrix_path 'distance_array_IL.npy' --dates_file_path  'Dates_IL.csv' --plot_type 'spiral' --aggregate_data --transform --threshold 600` 


## Installation

1. Download the repository as a ZIP file from GitHub:
https://github.com/thushari93/Spiral_Chronogram

2. Extract the ZIP file.

3. Navigate to the extracted folder:
```bash
cd <path-to-extracted-folder>
```

5. Install the necessary dependencies: 
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Import the `plot_chronogram` function and call it with the necessary parameters.

```python
from chronogram import plot_chronogram

plot_chronogram('distance_matrix.npy', 'dates.csv')
```
## Contributing

Feel free to submit issues to mho1@illinois.edu, fork the repository, and send pull requests!

## License

This project is licensed under the Creative Commons License.
