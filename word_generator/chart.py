import matplotlib.pyplot as plt
from torch import Tensor


def plot2d_embedded_characters(EMBEDDING_TABLE: Tensor, itos: dict, **kwargs):

    plt.figure(figsize=(8, 8))

    plt.scatter(EMBEDDING_TABLE[:, 0].detach(), EMBEDDING_TABLE[:, 1].detach())

    for i in range(EMBEDDING_TABLE.shape[0]):
        plt.text(
            EMBEDDING_TABLE[i, 0].item(),
            EMBEDDING_TABLE[i, 1].item(),
            itos[i],
            fontsize=12,
        )

    xlabel = kwargs.get("xlabel", "Embedding Dimension 1")
    ylabel = kwargs.get("ylabel", "Embedding Dimension 2")
    title = kwargs.get("title", "Character Embedding Space")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)

    plt.show()
