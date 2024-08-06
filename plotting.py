import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import math

def plot_distance_matrix_upper_triangle(distance_matrix_path, dates_file_path, color, date_type, plot_title=None):
    """
    Plot the upper triangle of the distance matrix as a heatmap.

    Parameters:
    distance_matrix_path (str): Path to the distance matrix file.
    dates_file_path (str): Path to the dates file.
    color (str): Color map for the heatmap.
    """
    dates_df, distance_matrix = check_and_load_files(distance_matrix_path, dates_file_path)
    dates_df['Date'] = pd.to_datetime(dates_df['Date'])
    if date_type == 'D':
        distance_df = pd.DataFrame(distance_matrix, index=dates_df['Date'].dt.date)
        distance_df.columns = dates_df['Date'].dt.date
    if date_type == 'M':
        distance_df = pd.DataFrame(distance_matrix, index=dates_df['Date'].dt.to_period('M'))
        distance_df.columns = dates_df['Date'].dt.to_period('M')
    if date_type == 'Y':
        distance_df = pd.DataFrame(distance_matrix, index=dates_df['Date'].dt.year)
        distance_df.columns = dates_df['Date'].dt.year
    mask = np.tril(np.ones_like(distance_matrix))
    fig, ax = plt.subplots(figsize=(9, 7))
    ax = sns.heatmap(distance_df, mask=mask, cmap=color)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.xaxis.set_label_position('top')
    plt.xticks(rotation=90)

    # add title if provided 
    if plot_title:
        plt.title(f'Upper Triangle Plot for {plot_title}')
        plt.savefig(f'upper_triangle_{plot_title}.png')
    else:
        plt.savefig('upper_triangle.png')
    plt.show()

def triangular_choronogram(df_plotting, color, date_type, plot_title=None):
    """
    Plot a triangular chronogram.

    Parameters:
    df_plotting (pd.DataFrame): DataFrame containing the plotting data.
    color (str): Color map for the plot.
    """
    df_plotting['Date'] = pd.to_datetime(df_plotting['Date'])
    if date_type == 'D':
        df_plotting["Date"] = df_plotting['Date'].dt.date
    if date_type == 'M':
        df_plotting["Date"] = df_plotting['Date'].dt.to_period('M').dt.to_timestamp()
    if date_type == 'Y':
        df_plotting["Date"] = df_plotting['Date'].dt.year

    fig, scatter_plot = plt.subplots(figsize=(9, 7))

    # Create a scatter plot with Date on x-axis, Lag on y-axis, and Dist represented by color
    scatter_plot = plt.scatter(x=df_plotting["Date"], y=df_plotting["Lag"], s=1, c=df_plotting["Dist"], cmap=color)

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45)

    # Label the x-axis
    plt.xlabel('Date')

    # Label the y-axis
    plt.ylabel('Time lag')

    # Add a color bar to the plot to represent the Distance
    plt.colorbar(scatter_plot, label='Distance')

    # add title if provided 
    if plot_title:
        plt.title(f'Triangular Chronogram for {plot_title}')
        plt.savefig(f'triangular_chronogram_{plot_title}.png')
    else:
        plt.savefig('triangular_chronogram.png')
    # Display the plot
    plt.show()

def spiral_chronogram(df_plotting, threshold, pattern, transform, gap, color, tick_step, date_type, plot_title=None):
    """
    Plot a spiral chronogram.

    Parameters:
    df_plotting (pd.DataFrame): DataFrame containing the plotting data.
    threshold (int): Threshold for the spiral.
    pattern (str): Pattern indicating overlap or non-overlap.
    transform (bool): Whether to apply log transformation to angles.
    gap (int): Gap for the spiral.
    color (str): Color map for the plot.
    tick_step (int): the highlight dots.
    date_type (str): define the resolution of date
    """

    # calculate 'Lag_first' which will be used as date index for plotting
    # df_plotting = df_plotting.sort_values(by='Date')
    df_plotting['Date'] = pd.to_datetime(df_plotting['Date'])
    if date_type == 'Y':
        # the format in the df_plotting['Date'] should be 'YYYY'
        df_plotting['Lag_first'] = (((df_plotting['Date'] - df_plotting['Date'].min()).dt.days)/365).astype(int)
        min_date = df_plotting['Date'].dt.year.astype(int).min()
        max_date = df_plotting['Date'].dt.year.astype(int).max()
        date_labels = pd.Series(range(min_date, max_date+1))
    elif date_type == 'M':
        # the format in the df_plotting['Date'] should be 'YYYY-MM'
        Date_temp = pd.to_datetime(df_plotting['Date'], format='%Y-%m')
        min_date = Date_temp.min()
        df_plotting['Lag_first'] = Date_temp.apply(lambda x: (x.year - min_date.year) * 12 + x.month - min_date.month)
        date_labels = pd.date_range(start=min_date, periods=df_plotting['Lag_first'].max(), freq='MS')
        date_labels = date_labels.strftime('%Y-%m')
    elif date_type == 'D':
        # the format in the df_plotting['Date'] should be 'YYYY-MM-DD'
        df_plotting['Lag_first'] = (pd.to_datetime(df_plotting['Date']) - pd.to_datetime(df_plotting['Date']).min()).dt.days
        date_labels = pd.date_range(start=pd.to_datetime(df_plotting['Date']).min(), end=pd.to_datetime(df_plotting['Date']).max(), freq='D')
        date_labels = date_labels.strftime('%Y-%m-%d')
    
    # transformed version - one full circle
    if transform == True:
        # create angle
        theta_values = (df_plotting['Lag_first'] / (df_plotting['Lag_first'].max()+1)) * 2 * np.pi
        theta = np.log(np.abs(theta_values) + 1) * np.sign(theta_values) 
        all_theta = (theta / (np.log(2*np.pi+1))) * 2 * np.pi
        # create radius
        all_r = df_plotting['Lag_first'] - df_plotting['Lag'] + gap
        # create color
        all_color = df_plotting['Dist']
    elif transform == False:
        # determine number of circles and the number in the last circle
        num_circles = math.ceil((df_plotting['Lag_first'].max()+1) / threshold)
        num_last = df_plotting['Lag_first'].max() + 1 - (num_circles-1)*threshold
        # determine colors representing difference
        # colors = df_plotting['Dist']
        # create angle and radius
        all_theta = []
        all_r = []
        all_color = []
        for i in range(num_circles):
            # subset dataframe for current circle
            start_index = threshold * i
            end_index = start_index+threshold-1
            df_name = f'df_circle_{i+1}'
            df_to_save = df_plotting.loc[(df_plotting['Lag_first'] >= start_index) & (df_plotting['Lag_first'] <= end_index)].reset_index(drop=True)
            globals()[df_name] = df_to_save
            color_df = df_to_save['Dist'] # determine colors representing difference
    
            # create angle
            Lag_first_diff = (df_to_save['Lag_first'] - threshold*i).astype(int)
            theta_values = (Lag_first_diff / threshold) * 2 * np.pi

            # create radius
            if pattern == 'overlap':
                r_values = df_to_save['Lag_first'] - df_to_save['Lag'] + gap
            elif pattern == 'nonoverlap':
                r_values = df_to_save['Lag_first'] - df_to_save['Lag'] + gap
                growth_rate = 0.0095
                if date_type == 'Y':
                    growth_rate = 0.012
                indices = np.arange(threshold)
                exponential_sequence = gap * (np.exp(growth_rate * indices) - 1)
                normalized_sequence = exponential_sequence / exponential_sequence.max()
                scaled_sequence = normalized_sequence * gap
                r_values += scaled_sequence[Lag_first_diff]
                r_values += gap*i
    
                if i>0:
                    for itr in range(i):
                        j = i-itr
                        r_gap = globals()[f'df_circle_{j}']['Lag_first'].min() + Lag_first_diff
                        r_values = r_values + r_gap

            # add to all_theta and all_r
            all_theta.append(theta_values)
            all_r.append(r_values)
            all_color.append(color_df)
    
        all_theta = np.concatenate(all_theta)
        all_r = np.concatenate(all_r)
        all_color = np.concatenate(all_color)
    
    # show plot
    x = all_r * np.cos(all_theta)
    y = all_r * np.sin(all_theta)
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = plt.scatter(x, y, c=all_color, cmap='Spectral', s=2)  # Color by distance value

    # Add boundary line and highlight tick
    
    if transform == True:
        theta_boundary = np.arange(0, df_plotting['Lag_first'].max()+1, 1)/(df_plotting['Lag_first'].max()+1) * 2 * np.pi
        theta_boundary_log = np.log(np.abs(theta_boundary) + 1) * np.sign(theta_boundary) 
        theta_boundary = (theta_boundary_log / (np.log(2*np.pi+1))) * 2 * np.pi
        r_boundary = np.arange(0, df_plotting['Lag_first'].max()+1, 1) + 1 +gap
        
    elif transform == False:
        # boundary angle
        theta_boundary = np.arange(0, threshold, 1)/threshold * 2 * np.pi
        theta_boundary = np.tile(theta_boundary, (num_circles-1))
        theta_boundary = np.concatenate((theta_boundary, np.arange(0, num_last, 1)/threshold * 2 * np.pi))
        # boundary radius
        if pattern == 'overlap':
            r_boundary = np.array([])
            for i in range(num_circles):
                if i < num_circles - 1:
                    radius_net = np.arange(0, threshold, 1)
                else:
                    radius_net = np.arange(0, num_last, 1)
                r_boundary_add = radius_net + threshold*i + gap
                r_boundary_add = r_boundary_add.astype(np.float64)
                r_boundary = np.concatenate((r_boundary, r_boundary_add))
        elif pattern == 'nonoverlap':
            r_boundary = np.array([])
            for i in range(num_circles):
                if i < num_circles - 1:
                    radius_net = np.arange(0, threshold, 1)
                else:
                    radius_net = np.arange(0, num_last, 1)
                r_boundary_add = radius_net + threshold*i + gap
                r_boundary_add = r_boundary_add.astype(np.float64)
                growth_rate = 0.0095
                if date_type == 'Y':
                    growth_rate = 0.012
                indices = np.arange(threshold)
                exponential_sequence = gap * (np.exp(growth_rate * indices) - 1)
                normalized_sequence = exponential_sequence / exponential_sequence.max()
                scaled_sequence = normalized_sequence * gap
                r_boundary_add += scaled_sequence[:len(radius_net)]
                r_boundary_add += gap*i
                if i>0:
                    for itr in range(i):
                        r_gap = radius_net + threshold * itr
                        r_boundary_add = r_boundary_add + r_gap
                r_boundary = np.concatenate((r_boundary, r_boundary_add))
    
    line_x = r_boundary * np.cos(theta_boundary)
    line_y = r_boundary * np.sin(theta_boundary)
    ax.plot(line_x, line_y, c='blue', linewidth=1)
    # highlight tick
    highlight_x = line_x[::tick_step]
    highlight_y = line_y[::tick_step]
    date_labels = date_labels[::tick_step]
    ax.scatter(highlight_x, highlight_y, s=20, c='black')
    # labels
    for i, label in enumerate(date_labels):
        ax.annotate(label, (highlight_x[i], highlight_y[i]), textcoords="offset points", xytext=(0,10), ha='center')
        #ax.annotate(label.strftime('%Y-%m-%d'), (highlight_x[i], highlight_y[i]), textcoords="offset points", xytext=(0,10), ha='center')

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Distance')

    ax.set_frame_on(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    # add title if provided 
    if plot_title:
        plt.title(f'Spiral Chronogram for {plot_title}')
        plt.savefig(f'spiral_chronogram_{plot_title}.png')
    else:
        plt.savefig('spiral_chronogram.png')
    plt.show()
    
def spiral_chronogram_3D(df_plotting, threshold, pattern, transform, gap, color, tick_step, date_type, plot_title=None):
    """
    Plot a spiral chronogram.

    Parameters:
    df_plotting (pd.DataFrame): DataFrame containing the plotting data.
    threshold (int): Threshold for the spiral.
    pattern (str): Pattern indicating overlap or non-overlap.
    transform (bool): Whether to apply log transformation to angles.
    gap (int): Gap for the spiral.
    color (str): Color map for the plot.
    tick_step (int): the highlight dots
    date_type (str): define the resolution of date
    """

    df_plotting['Date'] = pd.to_datetime(df_plotting['Date'])
    if date_type == 'Y':
        # the format in the df_plotting['Date'] should be 'YYYY'
        df_plotting['Lag_first'] = (((df_plotting['Date'] - df_plotting['Date'].min()).dt.days)/365).astype(int)
        min_date = df_plotting['Date'].dt.year.astype(int).min()
        max_date = df_plotting['Date'].dt.year.astype(int).max()
        date_labels = pd.Series(range(min_date, max_date+1))
    elif date_type == 'M':
        # the format in the df_plotting['Date'] should be 'YYYY-MM'
        Date_temp = pd.to_datetime(df_plotting['Date'], format='%Y-%m')
        min_date = Date_temp.min()
        df_plotting['Lag_first'] = Date_temp.apply(lambda x: (x.year - min_date.year) * 12 + x.month - min_date.month)
        date_labels = pd.date_range(start=min_date, periods=df_plotting['Lag_first'].max(), freq='MS')
        date_labels = date_labels.strftime('%Y-%m')
    elif date_type == 'D':
        # the format in the df_plotting['Date'] should be 'YYYY-MM-DD'
        df_plotting['Lag_first'] = (pd.to_datetime(df_plotting['Date']) - pd.to_datetime(df_plotting['Date']).min()).dt.days
        date_labels = pd.date_range(start=pd.to_datetime(df_plotting['Date']).min(), end=pd.to_datetime(df_plotting['Date']).max(), freq='D')
        date_labels = date_labels.strftime('%Y-%m-%d')
    
    # transformed version - one full circle
    if transform == True:
        # create angle
        theta_values = (df_plotting['Lag_first'] / (df_plotting['Lag_first'].max()+1)) * 2 * np.pi
        theta = np.log(np.abs(theta_values) + 1) * np.sign(theta_values) 
        all_theta = (theta / (np.log(2*np.pi+1))) * 2 * np.pi
        # create radius
        all_r = df_plotting['Lag_first'] - df_plotting['Lag'] + gap
        # create height
        all_z = all_theta
        # create color
        all_color = df_plotting['Dist']
    elif transform == False:
        # determine number of circles and the number in the last circle
        num_circles = math.ceil((df_plotting['Lag_first'].max()+1) / threshold)
        num_last = df_plotting['Lag_first'].max() + 1 - (num_circles-1)*threshold
        # determine colors representing difference
        #colors = df_plotting['Dist']
    
        all_theta = []
        all_r = []
        all_z = []
        all_color = []
        for i in range(num_circles):
            # subset dataframe for current circle
            start_index = threshold * i
            end_index = start_index+threshold-1
            df_name = f'df_circle_{i+1}'
            df_to_save = df_plotting.loc[(df_plotting['Lag_first'] >= start_index) & (df_plotting['Lag_first'] <= end_index)].reset_index(drop=True)
            globals()[df_name] = df_to_save
            color_df = df_to_save['Dist']
            # create angle
            Lag_first_diff = (df_to_save['Lag_first'] - threshold*i).astype(int)
            theta_values = (Lag_first_diff / threshold) * 2 * np.pi
            theta_values.reset_index(drop=True, inplace=True)
    
            # create z dimension
            z_values = theta_values + np.pi * 2 * i
    
            # create radius
            if pattern == 'overlap':
                r_values = df_to_save['Lag_first'] - df_to_save['Lag'] + gap
            elif pattern == 'nonoverlap':
                r_values = df_to_save['Lag_first'] - df_to_save['Lag'] + gap
                growth_rate = 0.0095
                if date_type == 'Y':
                    growth_rate = 0.012
                indices = np.arange(threshold)
                exponential_sequence = gap * (np.exp(growth_rate * indices) - 1)
                normalized_sequence = exponential_sequence / exponential_sequence.max()
                scaled_sequence = normalized_sequence * gap
                r_values += scaled_sequence[Lag_first_diff]
                r_values += gap*i
    
                if i>0:
                    for itr in range(i):
                        j = i-itr
                        r_gap = globals()[f'df_circle_{j}']['Lag_first'].min() + Lag_first_diff
                        r_values = r_values + r_gap

            # add to all_theta and all_r, all_z
            all_theta.append(theta_values)
            all_r.append(r_values)
            all_z.append(z_values)
            all_color.append(color_df)
    
        all_theta = np.concatenate(all_theta)
        all_r = np.concatenate(all_r)
        all_z = np.concatenate(all_z)
        all_color = np.concatenate(all_color)
    
    # Data points text label
    hover_dates = df_plotting['Date'].astype(str).tolist()
    hover_lags = df_plotting['Lag'].astype(str).tolist()
    hover_dists = df_plotting['Dist'].astype(str).tolist()

    # Boundary and highlight data
    if transform == True:
        theta_boundary = np.arange(0, df_plotting['Lag_first'].max()+1, 1)/(df_plotting['Lag_first'].max()+1) * 2 * np.pi
        theta_boundary_log = np.log(np.abs(theta_boundary) + 1) * np.sign(theta_boundary) 
        theta_boundary = (theta_boundary_log / (np.log(2*np.pi+1))) * 2 * np.pi
        r_boundary = np.arange(0, df_plotting['Lag_first'].max()+1, 1) + 1 + gap
        z_boundary = theta_boundary
    elif transform == False: 
        theta_boundary = np.arange(0, threshold, 1)/threshold * 2 * np.pi
        theta_boundary = np.tile(theta_boundary, (num_circles-1))
        theta_boundary = np.concatenate((theta_boundary, np.arange(0, num_last, 1)/threshold * 2 * np.pi))
    
        z_boundary = theta_boundary
        if num_circles > 0:
            for i in range(num_circles-1):
                z_boundary[threshold*(i+1):] += 2*np.pi
    
        # boundary radius
        if pattern == 'overlap':
            r_boundary = np.array([])
            for i in range(num_circles):
                if i < num_circles - 1:
                    radius_net = np.arange(0, threshold, 1)
                else:
                    radius_net = np.arange(0, num_last, 1)
                r_boundary_add = radius_net + threshold*i + gap
                r_boundary_add = r_boundary_add.astype(np.float64)
                r_boundary = np.concatenate((r_boundary, r_boundary_add))
        elif pattern == 'nonoverlap':
            r_boundary = np.array([])
            for i in range(num_circles):
                if i < num_circles - 1:
                    radius_net = np.arange(0, threshold, 1)
                else:
                    radius_net = np.arange(0, num_last, 1)
                r_boundary_add = radius_net + threshold*i + gap
                r_boundary_add = r_boundary_add.astype(np.float64)
                growth_rate = 0.0095
                if date_type == 'Y':
                    growth_rate = 0.012
                indices = np.arange(threshold)
                exponential_sequence = gap * (np.exp(growth_rate * indices) - 1)
                normalized_sequence = exponential_sequence / exponential_sequence.max()
                scaled_sequence = normalized_sequence * gap
                r_boundary_add += scaled_sequence[:len(radius_net)]
                r_boundary_add += gap*i
                if i>0:
                    for itr in range(i):
                        r_gap = radius_net + threshold * itr
                        r_boundary_add = r_boundary_add + r_gap
                r_boundary = np.concatenate((r_boundary, r_boundary_add))
        r_boundary = r_boundary + gap*0.2

    # 3D plot process
    plotly_data = go.Scatter3d(
        x=all_r * np.sin(all_theta),
        y=all_r * np.cos(all_theta),
        z=all_z,
        mode='markers',
        marker=dict(
            size=1,
            color=all_color,
            colorscale=color,
            opacity=0.8
        ),
        text=[f'Date: {d}<br>Lag: {l}<br>Dist: {dist}' for d, l, dist in zip(hover_dates, hover_lags, hover_dists)],
        hovertemplate='%{text}<extra></extra>'
    )

    # Add boundary line data
    boundary_line_data = go.Scatter3d(
        x=r_boundary * np.sin(theta_boundary),
        y=r_boundary * np.cos(theta_boundary),
        z=z_boundary,
        mode='lines',
        line=dict(
            color='blue',
            width=1
        )
    )

    highlight_x = (r_boundary[::tick_step] * np.sin(theta_boundary[::tick_step])).tolist()
    highlight_y = (r_boundary[::tick_step] * np.cos(theta_boundary[::tick_step])).tolist()
    highlight_z = z_boundary[::tick_step].tolist()
    date_labels_highlight = date_labels[::tick_step].tolist()

    # Add highlight data
    plotly_highlight_data = go.Scatter3d(
        x=highlight_x,
        y=highlight_y,
        z=highlight_z,
        mode='markers+text',
        marker=dict(
            size=5,
            color='black',
            opacity=1.0
        ),
        text=date_labels_highlight,
        textposition='top center'
    )

    plotly_layout = go.Layout(
        scene=dict(
            xaxis=dict(title=''),
            yaxis=dict(title=''),
            zaxis=dict(title='')
        ),
        title='Interactive 3D Spiral Plot'
    )
    # Save as html file
    plotly_fig = go.Figure(data=[plotly_data, boundary_line_data, plotly_highlight_data], layout=plotly_layout)
    
    # add title if provided 
    if plot_title:
        plotly_fig.write_html(f'interactive_3d_spiral_plot_{plot_title}.html')
    else:
        plotly_fig.write_html('interactive_3d_spiral_plot.html')

    # 2D plot process
    fig = plt.figure(figsize = (12,12))
    ax = fig.add_subplot(111, projection='3d')

    # Create scatter plot
    sc = ax.scatter(
        all_r * np.sin(all_theta),
        all_r * np.cos(all_theta),
        all_z,
        c=all_color,  # Color based on distance
        cmap='Spectral',
        s=1,
        marker='o',
        alpha=0.8,
        zorder=1
    )

    # Add boundary line
    ax.plot(
        r_boundary * np.sin(theta_boundary),
        r_boundary * np.cos(theta_boundary),
        z_boundary,
        color='blue',
        linewidth=1,
        zorder=2
    )

    # Create highlight dots
    sc_highlight = ax.scatter(
        r_boundary[::tick_step] * np.sin(theta_boundary[::tick_step]),
        r_boundary[::tick_step] * np.cos(theta_boundary[::tick_step]),
        z_boundary[::tick_step],
        c='black',  # Different color for highlighted points
        s=50,     # Larger size for highlighted points
        edgecolors='k',  # Black edge color to make highlighted points stand out
        marker='o',
        alpha=1.0,
        zorder=2
    )

    # labels
    for i, label in enumerate(date_labels[::tick_step]):
        ax.text(
            (r_boundary[::tick_step][i]) * np.sin(theta_boundary[::tick_step][i]),
            (r_boundary[::tick_step][i]) * np.cos(theta_boundary[::tick_step][i]),
            z_boundary[::tick_step][i],
            f'{label}',  # Customize this label as needed
            color='blue',
            zorder=3,
            bbox=dict(facecolor='white', alpha=0.3),
            ha='center'
        )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Distance')

    #ax.set_frame_on(False)
    #ax.xaxis.set_visible(False)
    #ax.yaxis.set_visible(False)
    ax.view_init(elev=20, azim=30)

    # add title if provided 
    if plot_title:
        plt.title(plot_title)
        plt.savefig(f'static_3d_spiral_plot_wtick_{plot_title}.png', dpi=300)
        plt.show()
    else: 
        plt.savefig('static_3d_spiral_plot_wtick.png', dpi=300)
        plt.show()

