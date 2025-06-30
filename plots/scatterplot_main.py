import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from results import results

def plot_scatter_rates_horizontal(results, xtick_fs=10, legend_fs=8):
    methods = list(results.keys())
    models = list(next(iter(results.values())).keys())

    markers = ['o', 's', '^', 'D', 'P']  # one per method
    cmap = plt.get_cmap('tab10')
    colors = cmap(range(len(models)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    # Scatter plots
    for j, method in enumerate(methods):
        for i, model in enumerate(models):
            data = results[method][model]
            x = data["Avg Tokens"] / 1000
            axes[0].scatter(x, data["Success Rate"],
                            marker=markers[j], edgecolor='black',
                            facecolor=colors[i], s=80, zorder=2)
            axes[1].scatter(x, data["Correctness Rate"],
                            marker=markers[j], edgecolor='black',
                            facecolor=colors[i], s=80, zorder=2)

    # Axes labels
    axes[0].set_xlabel("Avg Tokens (k)")
    axes[0].set_ylabel("Success Rate (%)")
    axes[1].set_xlabel("Avg Tokens (k)")
    axes[1].set_ylabel("Correctness Rate (%)")

    # Grids behind
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis='y', linestyle='--', linewidth=0.5, zorder=0)
        # ax.set_xscale('log')

    # Legends: one for methods by shape, one for models by color
    method_handles = [
        Line2D([0], [0], marker=markers[i], color='black', linestyle='none', label=method)
        for i, method in enumerate(methods)
    ]
    model_handles = [
        Line2D([0], [0], marker='s', color=colors[i], linestyle='none', label=model)
        for i, model in enumerate(models)
    ]

    # Place legends to the right, stacked vertically
    axes[0].legend(handles=method_handles, title="Method (shape)",
                   fontsize=legend_fs, frameon=False,
                   loc='center left', bbox_to_anchor=(1.1, 0.5))
    axes[1].legend(handles=model_handles, title="Model (color)",
                   fontsize=legend_fs, frameon=False,
                   loc='center left', bbox_to_anchor=(1.1, 0.5))

    plt.tight_layout()
    fig.savefig("scatter_rates.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

# Generate the horizontal scatter figure
plot_scatter_rates_horizontal(results)
