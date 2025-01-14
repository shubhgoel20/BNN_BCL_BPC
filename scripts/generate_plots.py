import sys
import matplotlib.pyplot as plt
from reader import get_info

def generate_plot(experiment_name, skip_approaches):

    data_dict, approach_data = get_info(experiment_name)

    num_tasks = data_dict['N']

    skip_approaches = skip_approaches.split(",") if len(skip_approaches) else []
    data_dict["M"] -= len(skip_approaches)

    plots_per_row = 5
    rows = num_tasks // plots_per_row if num_tasks > plots_per_row else 1

    fig, axes = plt.subplots(rows, plots_per_row, figsize=(15, 5 * rows), squeeze=False)

    line_styles = ['-', '--', '-.', 'dotted']
    markers = ['o', 'D', '.', '*']

    for task_idx in range(num_tasks):
        row, col = divmod(task_idx, plots_per_row)
        ax = axes[row, col]
        for exp_idx, (key, experiment_data) in enumerate(approach_data.items()):
            if key in skip_approaches:
                continue
            times = []
            accuracies = []
            for time_step, accuracy in enumerate(experiment_data[:, task_idx]):
                if accuracy != 0:
                    times.append(time_step)
                    accuracies.append(accuracy)
            ax.plot(times, accuracies, linestyle=line_styles[exp_idx % len(line_styles)], marker=markers[exp_idx % len(markers)], label=key)
        
        ax.set_title(f'Task {task_idx}')
        if row == rows-1:
            ax.set_xlabel('Tasks')
        if col == 0:
            ax.set_ylabel('Accuracy')
        ax.set_xticks(range(experiment_data.shape[0]))
        ax.grid(True)

    # Hide unused subplots
    for i in range(num_tasks, rows * plots_per_row):
        row, col = divmod(i, plots_per_row)
        axes[row, col].axis('off')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=data_dict['M'])
    
    plt.tight_layout(h_pad=0, w_pad=1, rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(bottom=0.1)
    plot_name = f'plots/{experiment_name}.pdf'
    if len(skip_approaches):
        plot_name = f'plots/{experiment_name}_SKIP_{"_".join(skip_approaches)}.pdf'
    plt.savefig(plot_name, format='pdf')
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_plots.py <experiment_name>")
        sys.exit(1)
    generate_plot(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else "")