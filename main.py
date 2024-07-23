import argparse
from plotting import plot_distance_matrix_upper_triangle, triangular_choronogram, spiral_chronogram
from utils import create_final_df, aggregate

def main():
    parser = argparse.ArgumentParser(description="Generate chronograms and spiral chronograms.")
    
    parser.add_argument('distance_matrix_path', type=str, help="Path to the distance matrix .npy file")
    parser.add_argument('dates_file_path', type=str, help="Path to the dates .csv file")
  
    parser.add_argument('--plot_type', type=str, default='spiral', help='[triangular, spiral, upper_triangle]')
    parser.add_argument('--transform', action='store_true', help="Whether to apply log transformation to angles")
    parser.add_argument('--aggregate_data', action='store_true', help="Whether to aggregate the data before plotting")
    parser.add_argument('--color', type=str, default='Spectral', help='Color map for the plot.')
    
    parser.add_argument('--gap', type=int, default=100, help="Gap for the spiral")
    parser.add_argument('--circle_coverage', type=int, default=0, help="Circle coverage for spiral chronogram")
    parser.add_argument('--color_bar', type=str, default='Spectral_r', help="Color map for the chronograms")
    
    args = parser.parse_args()
    
    if (args.plot_type == 'spiral' or args.plot_type == 'triangular' ) :
        df_plotting = create_final_df(args.distance_matrix_path, args.dates_file_path)
      if args.aggregate_data:
            df_plotting = aggregate(df_plotting)
        if (args.plot_type == 'triangular'):
            triangular_choronogram(df_plotting, args.color)
         elif (args.plot_type == 'spiral'):
            spiral_chronogram(df_plotting, threshold = args.threshold, pattern = args.pattern, transform = args.transform, gap = args.gap, color = args.color)
    elif (args.plot_type == 'upper_triangle'):
        plot_distance_matrix_upper_triangle(args.distance_matrix_path, args.dates_file_path, color = args.color) 
    else:
        raise ValueError("Invalid plot type specified. Use 'triangular', 'spiral', or 'upper_triangle'.")

if __name__ == "__main__":
    main()
