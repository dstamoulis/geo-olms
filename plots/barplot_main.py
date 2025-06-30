import matplotlib.pyplot as plt
import numpy as np

from results import results

def plot_bar_rates(results, success_ylim=(10,90), correct_ylim=(20,80),
                   xtick_fs=12, legend_fs=11.5):
    methods = list(results.keys())
    models = list(next(iter(results.values())).keys())
    n_methods, n_models = len(methods), len(models)

    group_width = 0.8
    x_offset = 0.3
    text_y_offset = 5
    bar_width = group_width / n_models
    x = np.arange(n_methods)

    # Colors and hatches
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(n_models)]
    hatches = ['/', '-', '\\', '+', '|', 'x', '.']

    fig, axes = plt.subplots(3, 1, figsize=(12, 6.5), sharex=True)

    # SUCCESS subplot (top)
    ax = axes[0]
    for i, model in enumerate(models):
        vals = [results[m][model]["Success Rate"] for m in methods]
        ax.bar(x + i*bar_width, vals, bar_width,
               label=model, color=colors[i], hatch=hatches[i])
    ax.set_ylabel("Success Rate %", fontsize=xtick_fs)
    ax.set_ylim(success_ylim)
    ax.set_xticks(x + group_width/2 - bar_width/2)
    # ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=xtick_fs)
    # ax.xaxis.tick_top()
    ax.tick_params(labelbottom=True)
    ax.set_axisbelow(True)
    ax.grid(axis='y', linestyle='--', linewidth=0.5)
    # Averages and text
    for j, m in enumerate(methods):
        avg = np.mean([results[m][mod]["Success Rate"] for mod in models])
        center = x[j]
        # left = center - group_width/2 + 0.05 + x_offset
        # right = center + group_width/2 - 0.05 + x_offset
        left = j + 0*bar_width - group_width/2 - 0.02 + x_offset
        right = j + n_models*bar_width - group_width/2 + 0.02 + x_offset
        # ax.hlines(avg, left, right, linestyles='dotted')
        # ax.text(j, avg + 1, f"Avg: {avg:.1f}%", ha='center', va='bottom', fontsize=10)
        ax.hlines(avg, left, right, linestyles='--', zorder=1)
        ax.text(center + x_offset, avg + text_y_offset, f"Avg: {avg:.1f}%",
                ha='center', va='bottom', fontsize=14,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1.0), zorder=3)

    # CORRECTNESS subplot (bottom)
    ax = axes[1]
    for i, model in enumerate(models):
        vals = [results[m][model]["Correctness Rate"] for m in methods]
        ax.bar(x + i*bar_width, vals, bar_width,
               label=model, color=colors[i], hatch=hatches[i])
    ax.set_ylabel("Correctness Rate %", fontsize=xtick_fs)
    ax.set_ylim(correct_ylim)
    ax.set_xticks(x + group_width/2 - bar_width/2)
    # ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=xtick_fs)
    ax.tick_params(labelbottom=True)
    ax.grid(axis='y', linestyle='--', linewidth=0.5)
    # Averages and text
    for j, m in enumerate(methods):
        avg = np.mean([results[m][mod]["Correctness Rate"] for mod in models])
        center = x[j]
        left = j + 0*bar_width - group_width/2 - 0.02 + x_offset
        right = j + n_models*bar_width - group_width/2 + 0.02 + x_offset
        # ax.hlines(avg, left, right, linestyles='dotted')
        # ax.text(j, avg + 1, f"Avg: {avg:.1f}%", ha='center', va='bottom', fontsize=10)
        ax.hlines(avg, left, right, linestyles='--', zorder=1)
        ax.text(center + x_offset, avg + text_y_offset, f"Avg: {avg:.1f}%",
                ha='center', va='bottom', fontsize=14,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1.0), zorder=3)


    # Tokens subplot (bottom)
    ax = axes[2]
    for i, model in enumerate(models):
        vals = [results[m][model]["Avg Tokens"]/1000 for m in methods]
        ax.bar(x + i*bar_width, vals, bar_width,
               label=model, color=colors[i], hatch=hatches[i])
    ax.set_ylabel("Avg Tokens $x 10^3$", fontsize=xtick_fs)
    # ax.set_ylim(correct_ylim)
    ax.set_xticks(x + group_width/2 - bar_width/2)
    # ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=xtick_fs)
    ax.grid(axis='y', linestyle='--', linewidth=0.5)
    # Averages and text
    for j, m in enumerate(methods):
        avg = np.mean([results[m][mod]["Avg Tokens"]/1000 for mod in models])
        center = x[j]
        left = j + 0*bar_width - group_width/2 - 0.02 + x_offset
        right = j + n_models*bar_width - group_width/2 + 0.02 + x_offset
        # ax.hlines(avg, left, right, linestyles='dotted')
        # ax.text(j, avg + 1, f"Avg: {avg:.1f}%", ha='center', va='bottom', fontsize=10)
        ax.hlines(avg, left, right, linestyles='--', zorder=1)
        ax.text(center + x_offset, avg + 0, f"Avg: {avg:.1f}k",
                ha='center', va='bottom', fontsize=14,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1.0), zorder=3)


    # Legend on top
    axes[0].legend(ncol=n_models, fontsize=legend_fs, frameon=False,
                   loc='upper center', bbox_to_anchor=(0.5, 1.25))

    plt.tight_layout()
    fig.savefig("barplot_main.png", dpi=300)
    plt.close(fig)

# Generate the updated plot
plot_bar_rates(results)
