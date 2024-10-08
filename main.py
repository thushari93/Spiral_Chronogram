import argparse
from plotting import plot_distance_matrix_upper_triangle, triangular_choronogram, spiral_chronogram, spiral_chronogram_3D
from utils import create_final_df
from utils import aggregate


def main():
    parser = argparse.ArgumentParser(description="Generate chronograms and spiral chronograms.")
    
    parser.add_argument('--distance_matrix_path', type=str, help="Path to the distance matrix .npy file")
    parser.add_argument('--dates_file_path', type=str, help="Path to the dates .csv file")
  
    parser.add_argument('--plot_type', type=str, default='spiral', help='[triangular, spiral, upper_triangle, spiral_3d]')
    parser.add_argument('--transform', action='store_true', help="Whether to apply log transformation to angles")
    parser.add_argument('--aggregate_data', action='store_true', help="Whether to aggregate the data before plotting")
    parser.add_argument('--color', type=str, default='Spectral', help='Color map for the plot.')
    parser.add_argument('--pattern', type=str, default='nonoverlap', help='Pattern indicating overlap or nonoverlap.')
    parser.add_argument('--gap', type=int, default=100, help="Gap for the spiral")
    
    parser.add_argument('--tick_step', type=int, default=365, help="the stepsize of highlight dots (only useful in spiral)")
    parser.add_argument('--date_type', type=str, default='D', help="Type of date ('D', 'M', 'Y')")
    parser.add_argument('--threshold', type=int, default=1000, help="Threshold for the spiral")
    parser.add_argument('--plot_title', type=str, default=None, help="Plot title")

    args = parser.parse_args()
    
    if (args.plot_type == 'spiral' or args.plot_type == 'triangular' or args.plot_type == 'spiral_3d') :
            df_plotting = create_final_df(args.distance_matrix_path, args.dates_file_path, args.date_type)
            if args.aggregate_data:
                    df_plotting = aggregate(df_plotting)
            if (args.plot_type == 'triangular'):
                    triangular_choronogram(df_plotting, args.color, args.date_type, args.plot_title)
            elif (args.plot_type == 'spiral'):
                    spiral_chronogram(df_plotting, args.threshold, args.pattern, args.transform, args.gap, args.color, args.tick_step, args.date_type, args.plot_title)
            elif (args.plot_type == 'spiral_3d'):
                    spiral_chronogram_3D(df_plotting, args.threshold, args.pattern, args.transform,args.gap, args.color, args.tick_step,args.date_type, args.plot_title)
    elif (args.plot_type == 'upper_triangle'):
        plot_distance_matrix_upper_triangle(args.distance_matrix_path, args.dates_file_path, args.color, args.date_type, args.plot_title) 
    else:
        raise ValueError("Invalid plot type specified. Use 'triangular', 'spiral', or 'upper_triangle' or 'spiral_3d'.")

if __name__ == "__main__":
    main()
