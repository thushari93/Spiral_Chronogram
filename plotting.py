import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math

def plot_distance_matrix_upper_triangle(distance_matrix_path, dates_file_path, color):
    """
    Plot the upper triangle of the distance matrix as a heatmap.

    Parameters:
    distance_matrix_path (str): Path to the distance matrix file.
    dates_file_path (str): Path to the dates file.
    color (str): Color map for the heatmap.
    """
    dates_df, distance_matrix = check_and_load_files(distance_matrix_path, dates_file_path)
    distance_df = pd.DataFrame(distance_matrix, index=dates_df['Date'])
    distance_df.columns = dates_df['Date']
    mask = np.triu(np.ones_like(distance_matrix))
    sns.heatmap(distance_df, mask=mask, cmap=color)
    plt.show()

def triangular_choronogram(df_plotting, color):
    """
    Plot a triangular chronogram.

    Parameters:
    df_plotting (pd.DataFrame): DataFrame containing the plotting data.
    color (str): Color map for the plot.
    """
    # Create a scatter plot with Date on x-axis, Lag on y-axis, and Dist represented by color
    scatter_plot = plt.scatter(x=df_plotting["Date"], y=df_plotting["Lag"], s=1, c=df_plotting["Dist"], cmap=color)

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45)

    # Label the x-axis
    plt.xlabel('Date')

    # Label the y-axis
    plt.ylabel('Time lag (days ago)')

    # Add a color bar to the plot to represent the Distance
    plt.colorbar(scatter_plot, label='Distance')

    # Display the plot
    plt.show()

def spiral_chronogram(df_plotting, threshold, pattern, transform, gap, color):
    """
    Plot a spiral chronogram.

    Parameters:
    df_plotting (pd.DataFrame): DataFrame containing the plotting data.
    threshold (int): Threshold for the spiral.
    pattern (str): Pattern indicating overlap or non-overlap.
    transform (bool): Whether to apply log transformation to angles.
    gap (int): Gap for the spiral.
    color (str): Color map for the plot.
    """

    df_plotting['Date'] = pd.to_datetime(df_plotting['Date'])
    df_plotting = df_plotting.sort_values(by='Date')

    df_plotting['Lag_first'] = (pd.to_datetime(df_plotting['Date']) - pd.to_datetime(df_plotting['Date']).min()).dt.days

    num_circles = math.floor((pd.to_datetime(df_plotting['Date']).max() - pd.to_datetime(df_plotting['Date']).min()).days // threshold) + 1
    colors = df_plotting['Dist']  # Color based on distance

    all_theta = []
    all_r = []
    for i in range(num_circles):
        start_date = pd.to_datetime(df_plotting['Date']).min() + pd.DateOffset(days=i * threshold)
        end_date = start_date + pd.DateOffset(days=threshold)
        df_name = f'df_circle_{i+1}'

        df_to_save = df_plotting.loc[(df_plotting['Date'] >= start_date) & (df_plotting['Date'] < end_date)].reset_index(drop=True)
        globals()[df_name] = df_to_save

        Lag_first_diff = (df_to_save['Lag_first'] - threshold*i).astype(int)

        theta_values = (Lag_first_diff / threshold) * 2 * np.pi
        if transform == True:
            theta = np.log(np.abs(theta_values) + 1) * np.sign(theta_values)  # Log transform
            theta_values = (theta / (np.log(2*np.pi+1))) * 2 * np.pi

        if pattern == 'overlap':
            r_values = df_to_save['Lag_first'] - df_to_save['Lag'] + gap
            # adjust the overlapped points at 0 and 2pi
            pi_multiples_indices = np.where(np.isclose(theta_values, 2 * np.pi))[0]
            zero_multiples_indices = np.where(np.isclose(theta_values, 0))[0]
            r_values[pi_multiples_indices] += r_values[zero_multiples_indices].max()
        elif pattern == 'nonoverlap':
            r_values = df_to_save['Lag_first'] - df_to_save['Lag'] + gap
            if transform == False:
                growth_rate = 0.0095
            if transform == True:
                growth_rate = 0.002
            indices = np.arange(threshold)
            exponential_sequence = gap * (np.exp(growth_rate * indices) - 1)
            normalized_sequence = exponential_sequence / exponential_sequence.max()
            scaled_sequence = normalized_sequence * gap
            r_values += scaled_sequence[Lag_first_diff]
            r_values += gap*i

            if i>0:
                for itr in range(i):
                    j = i-itr
                    base_date = globals()[f'df_circle_{j}']['Date'].min() + pd.to_timedelta(Lag_first_diff, unit='D')
                    r_gap = (base_date-df_plotting['Date'].min()).dt.days
                    r_gap.reset_index(drop=True, inplace=True)
                    r_values = r_values + r_gap

        all_theta.append(theta_values)
        all_r.append(r_values)

    all_theta = np.concatenate(all_theta)
    all_r = np.concatenate(all_r)

    x = all_r * np.cos(all_theta)
    y = all_r * np.sin(all_theta)

    fig, ax = plt.subplots(figsize=(9, 9))
    sc = ax.scatter(x, y, c=colors, cmap=color, s=1)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Distance')

    ax.set_frame_on(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    plt.show()
