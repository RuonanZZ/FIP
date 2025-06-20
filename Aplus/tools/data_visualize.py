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
    绘制表示数据分布的置信椭圆
    :param x: 数据的x坐标
    :param y: 数据的y坐标
    :param ax: matplotlib的Axes对象
    :param n_std: 标准差倍数，决定椭圆的大小
    :param facecolor: 椭圆的颜色
    :param kwargs: 传递给Ellipse的其他参数
    :return: 椭圆对象
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    # 计算均值和协方差矩阵
    cov = np.cov(x, y)
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # 计算椭圆的旋转角度和轴的长度
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    theta = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigvals)

    # 直接通过Ellipse的参数绘制置信椭圆
    ellipse = Ellipse((mean_x, mean_y), width=width, height=height, angle=theta,
                      facecolor=facecolor, **kwargs)

    ax.add_patch(ellipse)
    return ellipse

def random_index(data_len:int, sampling_rate=1.0, seed:int=None) -> list:
    """
    随机采样索引的函数

    参数:
    sample_size (int): 样本数量
    sampling_rate (float): 采样率，范围在(0, 1]
    seed (int or None): 随机数生成的种子，如果为None则不设置种子

    返回:
    list: 随机采样的索引列表
    """
    if not (0 < sampling_rate <= 1):
        raise ValueError("采样率必须在(0, 1]范围内")

    # 设置随机数种子
    np.random.seed(seed)

    # 计算需要采样的样本数量
    num_samples_to_select = int(data_len * sampling_rate)

    # 生成样本索引的随机排列
    all_indices = np.arange(data_len)
    np.random.shuffle(all_indices)

    # 从随机排列的索引中选择需要的数量
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
            # 统一转numpy
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
            # 统一转numpy
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
    绘制二维散点图，并添加置信椭圆

    参数:
    data_dict (dict): 包含多个维度为n乘2的numpy数组的字典
    add_lines (list): 设置要添加点间连线的key, 默认为空

    返回:
    None
    """
    plt.figure(figsize=[8, 6])
    plt.rcParams.update({
        'font.size': 28,  # 字体大小
        'font.family': 'serif',  # 字体类型
        'font.serif': ['Times New Roman'],  # 设置 Times New Roman 为全局字体
    })
    colors = ['#FFC000', '#7030A0', '#BFBFBF', '#d1b2e0', '#5C50FC']
    markers = ['s', '^', 'p']
    i = 0

    ax = plt.gca()  # 获取当前轴对象
    ax.set_facecolor('#F0F0F0')  # 设置背景颜色
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='white')  # 添加网格

    for label, data in data_dict.items():
        x, y = data[:, 0], data[:, 1]
        plt.scatter(x, y, label=label, c=colors[i], marker=markers[i])
        
        # 添加带有半透明填充的置信椭圆
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
    
    # 设置坐标轴范围和刻度间隔
    plt.ylim((-6, 9))
    plt.xlim((-7, 8))
    plt.xticks(ticks=[-7, -3, 0, 3, 7])  # 设置x轴刻度
    plt.yticks(ticks=[-5, 0, 5])  # 设置y轴刻度

    # 去除边框
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.show()

def plot_scatter_3d_from_dict(data_dict: dict, add_lines=[], epoch=1):
    """
    绘制三维散点图

    参数:
    data_dict (dict): 包含多个维度为n乘3的numpy数组的字典
    add_lines (list): 设置要添加点间连线的key, 默认为空

    返回:
    None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 遍历字典中的每个键值对
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
    绘制字典中多个1维NumPy数组的折线图，使用字典的键作为图例标签。

    参数:
    - data_dict: 包含多个1维/2维(label+数值) NumPy数组序列的字典。
    - conf_dict: 包含多个2维(上界+下界) NumPy数组序列的字典, key与data_dict相同。
    """
    # 创建一个新的图形
    plt.figure()

    # 遍历字典中的数据并绘制折线图
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
    # 添加图例
    plt.legend()
    # 添加轴标签和标题
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Line Chart')

    # 显示图形
    plt.show()

def plot_histogram_from_dict(data_dict, bins=10):
    """
    对字典中多个1维NumPy数组的数据进行统计分析，并绘制分布直方图，使用字典的键作为图例标签。

    参数:
    - data_dict: 包含多个1维NumPy数组的字典。
    - bins: 直方图的柱数，默认为10。
    """
    # 创建一个新的图形
    plt.figure()

    # 遍历字典中的数据并绘制分布直方图
    for key, values in data_dict.items():
        plt.hist(values, bins=bins, alpha=0.5, label=key)

    # 添加图例
    plt.legend()

    # 添加轴标签和标题
    plt.xlabel('value')
    plt.ylabel('count')
    plt.title('Histogram')

    # 显示图形
    plt.show()

def plot_box_chart_from_dict(data_dict: dict):
    """
    对字典中多个1维NumPy数组的数据进行统计分析，绘制箱型图，使用字典的键作为图例标签。

    参数:
    - data_dict: 包含多个1维NumPy数组的字典。
    """
    # 创建一个新的图形
    plt.figure()
    data_values = list(data_dict.values())
    plt.boxplot(data_values, labels=data_dict.keys())
    # 添加轴标签和标题
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Box Plot')

    # 显示图形
    plt.show()


# # ---------以下代码供测试-----------
# example_data_dict = {
#     'class_1': np.random.randn(1000, 20),
#     'class_2': np.random.randn(1000, 20)+1,
#     'class_3': np.random.randn(1000, 20)+2
# }
#
# dim_reducer = DimensionReducer(dim_origin=20, dim_target=3, method='pca')
# # 随机采样50%数据, 使用pca进行数据降维, 返回降维后的数据字典
# data_dict = dim_reducer.fit_transform(data=example_data_dict, sampling_rate=0.5)
# # 调用函数绘制三维散点图
# plot_scatter_3d_from_dict(data_dict)
# # 数据字典转换为pandas DataFrame
# print(data_dict_2_df(data_dict=data_dict, stack_dim=0))




