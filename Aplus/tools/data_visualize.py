import pandas as pd
import seaborn as sns
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    """
    Draw a confidence ellipse that represents the data distribution.

    :param x: X coordinates of the samples.
    :param y: Y coordinates of the samples.
    :param ax: Matplotlib Axes object.
    :param n_std: Number of standard deviations that determines the ellipse size.
    :param facecolor: Ellipse fill color.
    :param kwargs: Extra keyword arguments passed to `Ellipse`.
    :return: The created ellipse object.
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    # Compute the mean and covariance matrix.
    cov = np.cov(x, y)
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # Compute the ellipse rotation and axis lengths.
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    theta = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigvals)

    # Draw the ellipse directly from the Ellipse parameters.
    ellipse = Ellipse((mean_x, mean_y), width=width, height=height, angle=theta,
                      facecolor=facecolor, **kwargs)

    ax.add_patch(ellipse)
    return ellipse

def random_index(data_len:int, sampling_rate=1.0, seed:int=None) -> list:
    """
    Randomly sample indices from a dataset.

    Args:
        data_len (int): Number of samples.
        sampling_rate (float): Sampling rate in the range (0, 1].
        seed (int or None): Random seed. If None, the seed is not fixed.

    Returns:
        list: Sampled indices.
    """
    if not (0 < sampling_rate <= 1):
        raise ValueError("sampling_rate must be within the range (0, 1].")

    # Set the random seed.
    np.random.seed(seed)

    # Compute the number of samples to draw.
    num_samples_to_select = int(data_len * sampling_rate)

    # Generate a random permutation of sample indices.
    all_indices = np.arange(data_len)
    np.random.shuffle(all_indices)

    # Select the requested number of indices.
    selected_indices = all_indices[:num_samples_to_select]

    return selected_indices

class DimensionReducer:
    def __init__(self, dim_origin: int, dim_target: int, method='pca'):
        """
        Reduce data dimension for visualization.
        Args:
            dim_origin: dimension of origin data.
            dim_target: dimension after the reduction.
            method: [pca] or [tsne].
        """
        if method == 'pca':
            self.enbeder = PCA(n_components=dim_target)
        elif method == 'tsne':
            self.enbeder = TSNE(n_components=dim_target, init='pca', random_state=42)
            if dim_origin > 10:
                self.tsne_pca = PCA(n_components=10)
            else:
                self.tsne_pca = None
        else:
            raise ValueError("method='pca' or 'tsne'")

        self.method = method
        self.scaler = StandardScaler()
        self.fitted = False

    def fit_transform(self, data, sampling_rate=None, sampling_seed=None) -> dict:
        """

        Args:
            data: object or dict of numpy array or torch.Tensor with shape [batch, n_dim]
            sampling_rate: (0,1]
            sample_seed: default: None

        Returns:
            data or dict of numpy array, depends on input format
        """
        def _process(data, sampling_rate=None, sample_seed=None):
            # Convert inputs to NumPy in a consistent way.
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu()
                data = np.array(data)
            if sampling_rate is not None:
                sample_idx = random_index(data_len=len(data), sampling_rate=sampling_rate, seed=sample_seed)
                data = data[sample_idx]
            norm_data = self.scaler.fit_transform(data)
            if self.method == 'pca':
                data_result = self.enbeder.fit_transform(norm_data)
            elif self.method == 'tsne':
                if self.tsne_pca is not None:
                    norm_data = self.tsne_pca.fit_transform(norm_data)
                data_result = self.enbeder.fit_transform(norm_data)
            return data_result

        self.fitted = True

        if isinstance(data, dict):
            splits = [0]
            data_list = []
            for key in data.keys():
                _data = data[key]
                if isinstance(_data, torch.Tensor):
                    _data = _data.detach().cpu()
                    _data = np.array(_data)
                _data = _data.reshape(-1, _data.shape[-1])
                if sampling_rate:
                    reduced_idx = random_index(data_len=len(_data), sampling_rate=sampling_rate, seed=sampling_seed)
                    _data = _data[reduced_idx]
                data_list.append(_data)
                splits.append(len(_data)+splits[-1])

            data_all = np.concatenate(data_list, axis=0)
            data_all = _process(data=data_all)

            for i, key in enumerate(data.keys()):
                data[key] = data_all[splits[i]:splits[i+1]]
            return data
        else:
            return _process(data=data, sampling_rate=sampling_rate, sample_seed=sampling_seed)

    def transform(self, data, sampling_rate=None, sampling_seed=None) -> dict:
        """

        Args:
            data: object or dict of numpy array or torch.Tensor with shape [batch, n_dim]
            sampling_rate: (0,1]
            sample_seed: default: None

        Returns:
            data or dict of numpy array, depends on input format
        """
        if self.fitted == False:
            raise RuntimeError("call [fit_transform] in advance!")

        if self.method == 'tsne':
            raise RuntimeWarning("T-SNE refit everytime called, might result inconsistent representations!")
        def _process(data, sample_rate, sampling_seed):
            # Convert inputs to NumPy in a consistent way.
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu()
                data = np.array(data)
            if sampling_rate is not None:
                sample_idx = random_index(data_len=len(data), sampling_rate=sampling_rate, seed=sampling_seed)
                data = data[sample_idx]
            norm_data = self.scaler.transform(data)
            if self.method == 'pca':
                data_result = self.enbeder.transform(norm_data)
            elif self.method == 'tsne':
                if self.tsne_pca is not None:
                    norm_data = self.tsne_pca.transform(norm_data)
                data_result = self.enbeder.fit_transform(norm_data)

            return data_result

        if isinstance(data, dict):
            splits = [0]
            data_list = []
            for key in data.keys():
                _data = data[key]
                if isinstance(_data, torch.Tensor):
                    _data = _data.detach().cpu()
                    _data = np.array(_data)
                _data = _data.reshape(-1, _data.shape[-1])
                if sampling_rate:
                    reduced_idx = random_index(data_len=len(_data), sampling_rate=sampling_rate, seed=sampling_seed)
                    _data = _data[reduced_idx]
                data_list.append(_data)
                splits.append(len(_data) + splits[-1])

            data_all = np.concatenate(data_list, axis=0)
            data_all = _process(data=data_all)

            for i, key in enumerate(data.keys()):
                data[key] = data_all[splits[i]:splits[i + 1]]
            return data
        else:
            return _process(data=data, sampling_rate=sampling_rate, sampling_seed=sampling_seed)

def data_dict_2_df(data_dict: dict, stack_dim=0) -> pd.DataFrame:
    """
    Transform data dict to pandas DataFrames.
    Args:
        data_dict: {key_1: np.Array, key_2: np.Array, ...}
        stack_dim: How DataFrame stacked, choose [0] or [1]

    Returns:
        pandas DataFrames
    """
    import pandas as pd
    def _np2df(data, tag, stack_dim=stack_dim):
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        if stack_dim == 0:
            columns = [f'dim_{i}' for i in range(data.shape[1])]
        else:
            if data.shape[1] == 1:
                columns = [tag]
            else:
                columns = [f'{tag}_dim_{i}' for i in range(data.shape[1])]

        df = pd.DataFrame(data=data, columns=columns)

        if stack_dim == 0:
            df['tag'] = [tag for _ in range(len(data))]
        return df

    df_list = []
    for key in data_dict.keys():
        df_list.append(_np2df(data=data_dict[key], tag=key, stack_dim=stack_dim))
    data_df = pd.concat(df_list, axis=stack_dim)

    return data_df

def plot_scatter_2d_from_dict(data_dict: dict, add_lines=[], epoch=0):
    """
    Plot a 2D scatter chart and add confidence ellipses.
    """
    plt.figure(figsize=[8, 6])
    plt.rcParams.update({
        'font.size': 28,  # Font size.
        'font.family': 'serif',  # Font family.
        'font.serif': ['Times New Roman'],  # Use Times New Roman globally.
    })
    colors = ['#FFC000', '#7030A0', '#BFBFBF', '#d1b2e0', '#5C50FC']
    markers = ['s', '^', 'p']
    i = 0

    ax = plt.gca()  # Get the current axis.
    ax.set_facecolor('#F0F0F0')  # Set the background color.
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='white')  # Add grid lines.

    for label, data in data_dict.items():
        x, y = data[:, 0], data[:, 1]
        plt.scatter(x, y, label=label, c=colors[i], marker=markers[i])
        
        # Add a confidence ellipse with semi-transparent fill.
        confidence_ellipse(x, y, ax, n_std=2.0, facecolor=colors[i], edgecolor=colors[i], alpha=0.2, linewidth=5)
        
        if label in add_lines:
            for j in range(len(x) - 1):
                x_values = (x[j], x[j + 1])
                y_values = (y[j], y[j + 1])
                plt.plot(x_values, y_values, color='#00ff00')
        i += 1

    plt.xlabel('Dim1')
    plt.ylabel('Dim2')
    plt.legend(frameon=False, ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.18))  
    
    # Set axis ranges and tick intervals.
    plt.ylim((-6, 9))
    plt.xlim((-7, 8))
    plt.xticks(ticks=[-7, -3, 0, 3, 7])  # Set x-axis ticks.
    plt.yticks(ticks=[-5, 0, 5])  # Set y-axis ticks.

    # Hide the axes spines.
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.show()

def plot_scatter_3d_from_dict(data_dict: dict, add_lines=[], epoch=1):
    """
    Plot a 3D scatter chart.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Iterate over each entry in the dictionary.
    for label, data in data_dict.items():
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        ax.scatter(x, y, z, label=label)
        if label in add_lines:
            for i in range(len(x) - 1):
                x_values = (x[i], x[i + 1])
                y_values = (y[i], y[i + 1])
                z_values = (z[i], z[i + 1])
                plt.plot(x_values, y_values, z_values, color='#00ff00')

    ax.set_title("3D scatter")
    ax.legend()
    plt.show()
    # plt.savefig(f"image/{epoch}")
    # plt.close()

def plot_line_chart_from_dict(data_dict: dict, conf_dict=None):
    """
    Plot line charts for arrays stored in a dictionary, using keys as legend labels.
    """
    # Create a new figure.
    plt.figure()

    # Iterate over the dictionary and draw each line.
    for key, values in data_dict.items():
        if len(values.shape) == 1:
            values = values.reshape(-1, 1)

        if values.shape[1] == 2:
            x, y = values[:, 0], values[:, 1]
            plt.plot(x, y, label=key)
        else:
            plt.plot(values.reshape(-1), label=key)
        if conf_dict is not None:
            upper, lower = conf_dict[key][:, 0], conf_dict[key][:, 1]
            plt.fill_between(x, upper, lower, alpha=0.5)
    # Add the legend.
    plt.legend()
    # Add axis labels and the title.
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Line Chart')

    # Display the figure.
    plt.show()

def plot_histogram_from_dict(data_dict, bins=10):
    """
    Plot histograms for multiple 1D NumPy arrays stored in a dictionary.
    """
    # Create a new figure.
    plt.figure()

    # Iterate over the dictionary and draw each histogram.
    for key, values in data_dict.items():
        plt.hist(values, bins=bins, alpha=0.5, label=key)

    # Add the legend.
    plt.legend()

    # Add axis labels and the title.
    plt.xlabel('value')
    plt.ylabel('count')
    plt.title('Histogram')

    # Display the figure.
    plt.show()

def plot_box_chart_from_dict(data_dict: dict):
    """
    Plot box charts for multiple 1D NumPy arrays stored in a dictionary.
    """
    # Create a new figure.
    plt.figure()
    data_values = list(data_dict.values())
    plt.boxplot(data_values, labels=data_dict.keys())
    # Add axis labels and the title.
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Box Plot')

    # Display the figure.
    plt.show()


# # ---------Example code for testing---------
# example_data_dict = {
#     'class_1': np.random.randn(1000, 20),
#     'class_2': np.random.randn(1000, 20)+1,
#     'class_3': np.random.randn(1000, 20)+2
# }
#
# dim_reducer = DimensionReducer(dim_origin=20, dim_target=3, method='pca')
# # Randomly sample 50% of the data, reduce it with PCA, and return the reduced dictionary.
# data_dict = dim_reducer.fit_transform(data=example_data_dict, sampling_rate=0.5)
# # Call the helper to draw a 3D scatter plot.
# plot_scatter_3d_from_dict(data_dict)
# # Convert the data dictionary to a pandas DataFrame.
# print(data_dict_2_df(data_dict=data_dict, stack_dim=0))




