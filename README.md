# Chronogram Plotting Package

This package provides a function to plot a chronogram (triangular or spiral) based on a distance matrix and corresponding dates. The main function, `plot_chronogram`, generates these plots with customizable parameters. Here are some sample outputs generated using NCBI Virus SARS-CoV-2 Data.
Triangular Chronogram             |  Spiral Chronogram
:-------------------------:|:-------------------------:
 ![](https://github.com/thushari93/Spiral_Chronogram/blob/3fe56bc07996cc98743c7742b45c9981e49d9fe8/Images/triangular_chronogram.png)| ![](https://github.com/thushari93/Spiral_Chronogram/blob/4fc7e6adb0327be84c9745a6b4c729330223cf88/Images/spiral_chronogram.png) 

### Description

The `main.py` function plots a chronogram (triangular or spiral) based on the provided distance matrix and dates.

### Parameters

- `distance_matrix_path` (str): Path to the distance matrix file.
- `dates_file_path` (str): Path to the dates file.
- `threshold` (int, default=1000): Threshold for the spiral, number of date units in each circle.
- `pattern` (str, default='nonoverlap'): Pattern indicating overlap or non-overlap if spiral has more than one circle.
- `transform` (bool, default=False): Whether to apply log transformation to angles.
- `aggregate_data` (bool, default=False): Whether to aggregate the data before plotting.
- `gap` (int, default=100): Gap between circles for the spiral.
- `color` (str, default='Spectral'): Color map for the plot.
- `plot_type` (str, default='spiral'): Type of chronogram to plot ('triangular', 'spiral', 'upper_triangle', 'spiral_3d').
- `date_type` (str, default='D'): Unit of the dates ("D": days, "M": months, "Y": years).
- `tick_step` (int, default=100): The increment of highlight tick.

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
## Example Usage

Command to generate the plot under default settings;
```bash
python main.py --distance_matrix_path <path> --dates_file_path  <path> 
```

Command to generate a customized plot;
```bash
python main.py --distance_matrix_path 'distance_matrix_FL.npy' --dates_file_path  'Dates_FL.csv' --plot_type 'spiral' --aggregate_data --transform --threshold 600 
```

## Contributing

Feel free to submit issues to mho1@illinois.edu, fork the repository, and send pull requests!

## Aknowledgement
This project was funded by Institute for Mathematical and Statistical innovation (IMSI) and the University of Illinois Urbana-Champaign (UIUC) under the supervision of Dr. Wilson and Dr. Ho.

## License

This project is licensed under the Creative Commons License.
