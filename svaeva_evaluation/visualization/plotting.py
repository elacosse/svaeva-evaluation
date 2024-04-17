import base64
import io
from typing import List

import cv2
import matplotlib
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from svaeva_redux.schemas.redis import UserModel

RANDOM_SEED = 42


def transform_tSNE(arr: np.ndarray, n_components: int, perplexity: int) -> np.ndarray:
    """Transform the given ndarray using a tSNE model.

    Args:
        arr: np.ndarray. In this example, an embedding matrix (n by m), where n is the number of examples and m equals to the embedding dimension.
        n_components: int. The number of components for tSNE.
        perplexity: int. Perplexity for tSNE.

    Returns:
        vis_dims: np.ndarray. A transformed matrix of n by n_components.
    """
    tsne = TSNE(
        n_components=n_components, perplexity=perplexity, random_state=RANDOM_SEED, init="random", learning_rate=200
    )
    vis_dims = tsne.fit_transform(arr)
    return vis_dims


def plot_tSNE_embeddings(embed_arr_ls: List[np.ndarray], n_components: int, names: List[str], perplexity: int):
    """Plot transformed embedding vectors with predefined labels.

    Args:
        embed_arr_ls: a list of np.ndarray. Each np.ndarray is a matrix with embeddings corresponding to data examples.
        n_components: int. The number of components for tSNE.
        names: a list of str. The names of the data sources. The length of this list should be the same as the length of embed_arr_ls.
        perplexity: int. Perplexity for tSNE.
        save_path: path to save figure
    Returns:
        None
    """
    vis_dims = transform_tSNE(np.concatenate(embed_arr_ls), n_components, perplexity)
    colors = ["red", "blue", "green", "orange", "purple"]
    list_names_set = list(set(names))
    colormap = matplotlib.colors.ListedColormap(colors)
    color_indices = []
    # for label in range(len(embed_arr_ls)):
    for label in names:
        color_indices += [list_names_set.index(label)]  # [label] * len(embed_arr_ls[label])
    assert len(vis_dims) == len(color_indices)
    x = [x for x, y in vis_dims]
    y = [y for x, y in vis_dims]

    fig, ax = plt.subplots()
    ax.scatter(x, y, c=color_indices, cmap=colormap, alpha=0.3)
    # for label in range(len(embed_arr_ls)):
    for color_index in range(len(colors)):
        # color = colors[label]
        color = colors[color_index]
        label_indices = [i for i, value in enumerate(color_indices) if value == color_index]
        avg_x = np.array(x)[label_indices].mean()
        avg_y = np.array(y)[label_indices].mean()
        ax.scatter(avg_x, avg_y, marker="x", color=color, s=100, label=list_names_set[color_index])

    ax.legend()
    plt.title("Conversations sample data visualized in language using t-SNE")
    return fig


def plot_3d_pca_embeddings(users: List[UserModel], names: List[str]) -> plt.Figure:
    pca = PCA(n_components=3)
    matrix = np.array([np.array(user.conversation_embedding) for user in users])
    vis_dims = pca.fit_transform(matrix)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(projection="3d")
    cmap = plt.get_cmap("tab10")

    # Plot each sample category individually such that we can set label name.
    np_names = np.array(names)
    for i, cat in enumerate(list(set(names))):
        sub_matrix = vis_dims[np_names == cat]
        x = sub_matrix[:, 0]
        y = sub_matrix[:, 1]
        z = sub_matrix[:, 2]
        color = cmap(i)
        ax.scatter(x, y, zs=z, zdir="z", color=color, label=cat)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(bbox_to_anchor=(1.1, 1))

    return fig


def plot_edge_distribution(distances: List[dict]):
    """Plot the distribution of edge weights.

    Args:
        distances: List[dict]. A list of distances. Each element is a dictionary with the following keys:
            - "id": str. A cluster identifier for the node.
            - "distance": dict. A dictionary of distances to other nodes. The key is the node id and the value is the distance.

    Returns:
        None
    """
    fig = plt.figure()
    edge_weights = []
    for distance in distances:
        for key, value in distance["distance"].items():
            edge_weights.append(value)

    sns.histplot(edge_weights, kde=True)
    plt.title("Distribution of edge weights")
    plt.xlabel("Edge weight")
    plt.ylabel("Frequency")

    return fig


def return_circle_cropped_image_from_user(user: UserModel) -> Image.Image:
    """Return a circle cropped image from a user object"""
    image = Image.open(io.BytesIO(base64.b64decode(user.avatar_image_bytes)))
    image = np.array(image)
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), int(0.45 * h), 255, -1)
    result = cv2.bitwise_and(image, image, mask=mask)
    image = Image.fromarray(result)
    return image
