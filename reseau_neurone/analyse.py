import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_training_results(json_filepath):
    """
    Load training results from a JSON file into a pandas DataFrame.
    """
    with open(json_filepath, 'r') as file:
        data = json.load(file)
    df = pd.json_normalize(data)
    return df

def expand_config_parameters(df):
    """
    Expand the nested configuration parameters into separate columns.
    """
    # Expand dropout_rates and layer_sizes into separate columns
    dropout_df = df['config.dropout_rates'].apply(pd.Series)
    dropout_df = dropout_df.rename(columns=lambda x: f'dropout_rate_{x+1}')
    
    layer_df = df['config.layer_sizes'].apply(pd.Series)
    layer_df = layer_df.rename(columns=lambda x: f'layer_size_{x+1}')
    
    # Combine with the main DataFrame
    df = pd.concat([df, dropout_df, layer_df], axis=1)
    df = df.drop(['config.dropout_rates', 'config.layer_sizes'], axis=1)
    
    return df

def ensure_parameter_columns(df, parameter_columns):
    """
    Ensure all parameter columns exist in the DataFrame.
    If a column is missing, fill it with NaN.
    """
    for col in parameter_columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df

def plot_boxplots(df, parameter, metrics):
    """
    Plot boxplots for each metric against a specific parameter.
    """
    plt.figure(figsize=(12, 6))
    for metric in metrics:
        plt.subplot(1, len(metrics), metrics.index(metric)+1)
        sns.boxplot(x=parameter, y=metric, data=df)
        plt.title(f'{metric.capitalize()} vs {parameter.replace("_", " ").capitalize()}')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_scatter(df, parameter, metric):
    """
    Plot scatter plot of a parameter against a metric.
    """
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=parameter, y=metric, data=df)
    plt.title(f'{metric.capitalize()} vs {parameter.replace("_", " ").capitalize()}')
    plt.xlabel(parameter.replace("_", " ").capitalize())
    plt.ylabel(metric.capitalize())
    plt.show()

def plot_barplots(df, parameter, metrics):
    """
    Plot barplots showing the average metric values for each parameter value.
    """
    df_grouped = df.groupby(parameter)[metrics].mean().reset_index()
    df_melted = df_grouped.melt(id_vars=parameter, value_vars=metrics, var_name='Metric', value_name='Value')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=parameter, y='Value', hue='Metric', data=df_melted)
    plt.title(f'Average Metrics by {parameter.replace("_", " ").capitalize()}')
    plt.xticks(rotation=45)
    plt.ylabel('Average Value')
    plt.legend(title='Metrics')
    plt.tight_layout()
    plt.show()

def main():
    # Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, '..', 'results', 'training_results_20241218_022242.json')
    
    # Load training results
    df = load_training_results(json_path)
    print(f"Loaded {len(df)} training results.")
    
    # Expand configuration parameters
    df = expand_config_parameters(df)
    
    # Define metric and parameter columns
    metric_columns = ['metrics.loss', 'metrics.accuracy', 'metrics.auc']
    parameter_columns = ['config.learning_rate', 'config.batch_size'] + \
                        [f'dropout_rate_{i}' for i in range(1, 4)] + \
                        [f'layer_size_{i}' for i in range(1, 4)]
    
    # Ensure all parameter columns exist
    df = ensure_parameter_columns(df, parameter_columns)
    
    # Clean DataFrame: Convert data types
    for param in ['config.learning_rate', 'config.batch_size'] + \
                 [f'dropout_rate_{i}' for i in range(1, 4)] + \
                 [f'layer_size_{i}' for i in range(1, 4)]:
        if 'dropout_rate' in param:
            df[param] = pd.to_numeric(df[param], errors='coerce')
        elif 'layer_size' in param or param == 'config.batch_size':
            df[param] = pd.to_numeric(df[param], errors='coerce', downcast='integer')
        elif param == 'config.learning_rate':
            df[param] = pd.to_numeric(df[param], errors='coerce')
    
    # Drop rows with missing values in parameters or metrics
    df_clean = df.dropna(subset=parameter_columns + metric_columns)
    print(f"After cleaning, {len(df_clean)} training results remain.")
    
    # Analysis: Find Optimal Parameters
    # Example: Finding parameter combinations with highest accuracy
    top_n = 5
    top_models = df_clean.nlargest(top_n, 'metrics.accuracy')
    print(f"\nTop {top_n} Models by Accuracy:")
    print(top_models[['model_id'] + metric_columns + parameter_columns])
    
    # Visualization: Boxplots
    for param in ['config.learning_rate', 'config.batch_size']:
        plot_boxplots(df_clean, param, metric_columns)
    
    # Visualization: Dropout Rates
    for i in range(1, 4):
        param = f'dropout_rate_{i}'
        plot_boxplots(df_clean, param, metric_columns)
    
    # Visualization: Layer Sizes
    for i in range(1, 4):
        param = f'layer_size_{i}'
        plot_boxplots(df_clean, param, metric_columns)
    
    # Scatter Plots: Parameter vs Metrics
    for param in parameter_columns:
        for metric in metric_columns:
            plot_scatter(df_clean, param, metric)
    
    # Bar Plots: Average Metrics by Parameter
    for param in parameter_columns:
        plot_barplots(df_clean, param, metric_columns)
    
    # Correlation Analysis (Optional)
    correlations = df_clean[parameter_columns + metric_columns].corr()
    plt.figure(figsize=(14, 10))
    sns.heatmap(correlations, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix Between Parameters and Metrics')
    plt.show()

if __name__ == "__main__":
    main()