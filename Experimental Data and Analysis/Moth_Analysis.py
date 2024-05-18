# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 11:17:10 2024

@author: Thomas
"""

# Import required libraries
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from matplotlib.colors import LogNorm
from scipy.stats import ttest_ind

# Constants
DATA_PATTERN_COVERAGE = "Moth_Coverage_*.csv"
DATA_PATTERN_ACTIVE_SENSING = "Moth_Active_Sensing_*.csv"

# Function to read all CSV files for a given pattern and return a dictionary of DataFrames
def read_data(pattern):
    all_files = glob.glob(pattern)
    df_dict = {}
    for filename in all_files:
        # Extract the trial number from the filename
        trial_number = filename.split('_')[-1].split('.')[0]
        df = pd.read_csv(filename, index_col=None, header=0)
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        df_dict[trial_number] = df
    return df_dict

def plot_visit_heatmap(dfs_dict, agent_type, VMAX=None):
    # Initialize a dictionary to count visits for each location
    visit_counts = {}
    
    # Iterate over each trial DataFrame and tally visits
    for df in dfs_dict.values():
        for index, row in df.iterrows():
            loc = (row['x'], row['y'])
            if loc not in visit_counts:
                visit_counts[loc] = 1
            else:
                visit_counts[loc] += 1
                
    # Convert the visit counts to a DataFrame for plotting
    visit_df = pd.DataFrame(list(visit_counts.items()), columns=['Location', 'Count'])
    visit_df['x'], visit_df['y'] = zip(*visit_df['Location'])
    visit_counts_matrix = visit_df.pivot(index='y', columns='x', values='Count').fillna(0)
    
    # Create a heatmap using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(visit_counts_matrix.iloc[::-1], cmap='jet', annot=False,vmin=0, vmax=VMAX)
    plt.title(f'Visit Heatmap for {agent_type}')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.savefig(f'{agent_type}_visit_heatmap.pdf')
    plt.show()
    plt.close()  # Close the plot to free memory

def plot_terminal_state_heatmap(dfs_dict, agent_type, VMAX=None):
    # Initialize a dictionary to count occurrences of the terminal state for each location
    terminal_counts = {}
    
    # Iterate over each trial DataFrame and record the terminal state
    for df in dfs_dict.values():
        # The terminal state is located in the last row of each DataFrame
        terminal_loc = (df.iloc[-1]['x'], df.iloc[-1]['y'])
        if terminal_loc not in terminal_counts:
            terminal_counts[terminal_loc] = 1
        else:
            terminal_counts[terminal_loc] += 1

    # Convert the terminal counts to a DataFrame for plotting
    terminal_df = pd.DataFrame(list(terminal_counts.items()), columns=['Location', 'Count'])
    terminal_df['x'], terminal_df['y'] = zip(*terminal_df['Location'])
    terminal_counts_matrix = terminal_df.pivot(index='y', columns='x', values='Count').fillna(0)

    # Reindex the DataFrame to ensure it starts from (0,0) and goes up to the max x and y
    all_x = np.arange(0, terminal_counts_matrix.columns.max() + 1)
    all_y = np.arange(0, terminal_counts_matrix.index.max() + 1)
    terminal_counts_matrix = terminal_counts_matrix.reindex(index=all_y, columns=all_x, fill_value=0)


    # Create a heatmap using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(terminal_counts_matrix.iloc[::-1], cmap='jet', annot=False, vmin=0, vmax=VMAX)
    plt.title(f'Terminal State Heatmap for {agent_type}')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.savefig(f'{agent_type}_terminal_state_heatmap.pdf',dpi=300)
    plt.show()
    plt.close()  # Close the plot to free memory

def plot_combined_luminosity_heatmaps(dfs_coverage_dict, dfs_active_sensing_dict):
    # Combine all DataFrames from both dictionaries into a single DataFrame
    all_dfs = list(dfs_coverage_dict.values()) + list(dfs_active_sensing_dict.values())
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Calculate the average and variance of luminosity for each location
    lum_avg = combined_df.groupby(['x', 'y'])['lum'].mean().reset_index(name='average_lum')
    lum_var = combined_df.groupby(['x', 'y'])['lum'].var().reset_index(name='variance_lum')
    
    # Create pivot tables for the average and variance data
    lum_avg_matrix = lum_avg.pivot(index='y', columns='x', values='average_lum').fillna(0.0001)
    lum_var_matrix = lum_var.pivot(index='y', columns='x', values='variance_lum').fillna(0.0001)
    
    # Plot the average luminosity heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(lum_avg_matrix.iloc[::-1], cmap='inferno', annot=False)  # Reverse the y-axis
    plt.title('Combined Average Luminosity Heatmap')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.savefig('AVG_LUM.pdf', dpi = 300)
    plt.show()
    plt.close()  # Close the plot to free memory

    # Plot the variance of luminosity heatmap
    plt.figure(figsize=(10, 8))
    data = lum_var_matrix.iloc[::-1]
    # Ensure vmin and vmax are scalars
    vmin, vmax = data.min().min(), data.max().max()
    sns.heatmap(data, norm=LogNorm(vmin=vmin, vmax=vmax), cmap='inferno', annot=False)  # Reverse the y-axis
    plt.title('Combined Variance of Luminosity Heatmap')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.savefig('VAR_LUM.pdf', dpi=300)
    plt.show()
    plt.close()  # Close the plot to free memory




def calculate_performance_metrics(dfs_coverage_dict, dfs_active_sensing_dict, CI=0.99):
    def extract_metrics(df):
        # Extracts start and end times, calculates elapsed time and total timesteps
        if not df.empty:
            start_time = df.iloc[0]['time']
            end_time = df.iloc[-1]['time']
            elapsed_time = (end_time - start_time) / 1e9  # Convert nanoseconds to seconds
            total_timesteps = df.iloc[-1]['timestep']
            return elapsed_time, total_timesteps
        return 0, 0
    
    def calculate_stats(data):
        # Calculate mean, standard error and confidence interval
        mean_val = np.mean(data)
        std_error = stats.sem(data)
        conf_interval = stats.t.interval(CI, len(data)-1, loc=mean_val, scale=std_error)
        return mean_val, std_error, conf_interval

    # Extract and calculate metrics for both agent types
    coverage_time, coverage_timesteps = zip(*[extract_metrics(df) for df in dfs_coverage_dict.values()])
    active_time, active_timesteps = zip(*[extract_metrics(df) for df in dfs_active_sensing_dict.values()])

    # Calculate statistics for both agent types
    coverage_time_mean, coverage_time_sem, coverage_time_conf = calculate_stats(coverage_time)
    coverage_timesteps_mean, coverage_timesteps_sem, coverage_timesteps_conf = calculate_stats(coverage_timesteps)
    
    active_time_mean, active_time_sem, active_time_conf = calculate_stats(active_time)
    active_timesteps_mean, active_timesteps_sem, active_timesteps_conf = calculate_stats(active_timesteps)

   # Two-sided T-tests for H0
    time_t_result = ttest_ind(coverage_time, active_time )
    timesteps_t_result = ttest_ind(coverage_timesteps, active_timesteps)

    # One-sided T-tests for H1
    time_p_value_gretaer = ttest_ind(coverage_time, active_time, alternative='greater')
    timesteps_p_value_greater = ttest_ind(coverage_timesteps, active_timesteps, alternative='greater')
    time_p_value_less = ttest_ind(coverage_time, active_time, alternative='less')
    timesteps_p_value_less = ttest_ind(coverage_timesteps, active_timesteps, alternative='less')

    # Combine statistics and hypothesis test results into a DataFrame
    # Combine statistics and hypothesis test results into a DataFrame
    metrics = pd.DataFrame({
        'Agent Type': ['Coverage', 'Coverage', 'Active Sensing', 'Active Sensing'],
        'Performance Metric': ['Elapsed Time to Completion', 'Action Steps to Completion'] * 2,
        'Time(s) / Action Steps': [coverage_time_mean, coverage_timesteps_mean, active_time_mean, active_timesteps_mean],
        'Std Error': [coverage_time_sem, coverage_timesteps_sem, active_time_sem, active_timesteps_sem],
        'CI Lower': [coverage_time_conf[0], coverage_timesteps_conf[0], active_time_conf[0], active_timesteps_conf[0]],
        'CI Upper': [coverage_time_conf[1], coverage_timesteps_conf[1], active_time_conf[1], active_timesteps_conf[1]],
        'Two-Sided P-Value': [time_t_result.pvalue, timesteps_p_value_greater.pvalue] * 2,
        'One-Sided P-Value H1 Greater': [time_p_value_gretaer.pvalue, timesteps_p_value_greater.pvalue] * 2,
        'One-Sided P-Value H1 Less': [timesteps_p_value_less.pvalue, timesteps_p_value_less.pvalue] * 2,
        'T-Statistic': [time_t_result.statistic, timesteps_t_result.statistic] * 2,
        'Degrees of Freedom (df)': [time_t_result.df, timesteps_t_result.df] * 2
    })

    # Print metrics for verification
    print(metrics)

    return metrics

def plot_performance_bar_charts(dfs_coverage_dict, dfs_active_sensing_dict, CI=0.99):
   
    # Prepare data for plotting
    metrics = calculate_performance_metrics(dfs_coverage_dict, dfs_active_sensing_dict, CI )

    # Plotting
    plt.figure(figsize=(7.5, 3.3))
    bar_plot = sns.barplot(x='Performance Metric', y='Time(s) / Action Steps', hue='Agent Type', data=metrics, capsize=.1)

    # Find the x position for the error bars (reintroduced)
    bar_centers = [p.get_x() + p.get_width() / 2 for p in bar_plot.patches]

    # Calculate the errors as the distance from the mean to the confidence interval bounds
    metrics['CI Error Lower'] = metrics['Time(s) / Action Steps'] - metrics['CI Lower']
    metrics['CI Error Upper'] = metrics['CI Upper'] - metrics['Time(s) / Action Steps']
    ci_errors = metrics[['CI Error Lower', 'CI Error Upper']].to_numpy().T

    # Add error bars to the bar plot, correctly aligned with the bars
    for i, row in metrics.iterrows():
        bar_group_idx = i % 2  # 0 for the first group (Time), 1 for the second (Timesteps)
        position = bar_centers[i]  # bar_centers already takes into account the hue grouping
        plt.errorbar(position, row['Time(s) / Action Steps'], yerr=[[ci_errors[0, i]], [ci_errors[1, i]]], fmt='none', c='black', capsize=5)

    plt.title(f'Comparison of Moth Agent Performance.\nAverage Completion time and action steps with {CI} Confidence Interval')
    plt.savefig('Performance_Analysis.pdf', dpi=300)
    plt.show()
    plt.close()




# In the main execution block, read data into a dictionary of DataFrames
if __name__ == "__main__":
    # Dictionaries to store DataFrames for each trial
    df_coverage = read_data(DATA_PATTERN_COVERAGE)
    df_active_sensing = read_data(DATA_PATTERN_ACTIVE_SENSING)

    # Plot heatmaps for visit counts
    plot_visit_heatmap(df_coverage, "Coverage",20)
    plot_visit_heatmap(df_active_sensing, "Active Sensing",25)

    # Plot heatmaps for brightest locations
    plot_terminal_state_heatmap(df_coverage, "Coverage")
    plot_terminal_state_heatmap(df_active_sensing, "Active Sensing")

    # # Plot heatmaps for average and variance luminosity
    plot_combined_luminosity_heatmaps(df_coverage, df_active_sensing)
    
    plot_performance_bar_charts(df_coverage, df_active_sensing)
    # # Plot bar charts for performance with error bars
    
def analyze_luminosity_at_location(dfs_coverage_dict, dfs_active_sensing_dict, x, y):
    # Gather all luminosity values at the specified location across all trials for both agents
    lum_values = []
    
    # Extract values from coverage trials
    for df in dfs_coverage_dict.values():
        values = df[(df['x'] == x) & (df['y'] == y)]['lum'].tolist()
        lum_values.extend(values)
    
    # Extract values from active sensing trials
    for df in dfs_active_sensing_dict.values():
        values = df[(df['x'] == x) & (df['y'] == y)]['lum'].tolist()
        lum_values.extend(values)
    
    # Print the luminosity values
    print("Luminosity values at location (", x, ",", y, "):", lum_values)
    
    # Calculate mean, standard deviation, and variance
    mean_lum = np.mean(lum_values)
    std_lum = np.std(lum_values, ddof=1)  # Using Bessel's correction (ddof=1)
    var_lum = np.var(lum_values, ddof=1)  # Using Bessel's correction (ddof=1)
    
    # Print calculated statistics
    print("Mean luminosity:", mean_lum)
    print("Standard deviation:", std_lum)
    print("Variance:", var_lum)
    
    # Plot the distribution of luminosity values with a Gaussian fit
    plt.figure(figsize=(10, 6))
    sns.histplot(lum_values, kde=False, stat="density", bins=100, color='blue')
    
    # Fit and plot a normal distribution
    xmin, xmax = plt.xlim()
    x_axis = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x_axis, mean_lum, std_lum)
    plt.plot(x_axis, p, 'k', linewidth=2, color='red')
    
    title = f"Statistical Analysis of Luminosity Readings at {x},{y}\nFit results: mu = %.2f,  std = %.2f, var = %.2f" % (mean_lum, std_lum, var_lum)
    plt.title(title)
    plt.xlabel('Luminosity')
    plt.ylabel('Density')
    plt.savefig(f'LUM{x}{y}.pdf',dpi=300)
    plt.show()

# Example usage, assuming dfs_coverage_dict and dfs_active_sensing_dict are available:
analyze_luminosity_at_location(df_coverage, df_active_sensing, 3,7)
metrics = calculate_performance_metrics(df_coverage, df_active_sensing)

