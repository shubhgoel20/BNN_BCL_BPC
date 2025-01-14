import sys
import matplotlib.pyplot as plt
from reader import get_info

def generate_plot(experiment_name):

    data_dict, approach_data = get_info(experiment_name)

    # Determine the number of tasks (columns in the matrix)
    num_tasks = data_dict['N']

    # Plot configuration: Arrange up to 5 plots per row
    plots_per_row = 5
    rows = data_dict['M'] // plots_per_row

    fig, axes = plt.subplots(rows, plots_per_row, figsize=(15, 5 * rows), squeeze=False)

    line_styles = ['-', '--', '-.', 'dotted']
    markers = ['o', 'D', '.', '*']

    for task_idx in range(num_tasks):
        row, col = divmod(task_idx, plots_per_row)
        ax = axes[row, col]
        for exp_idx, (key, experiment_data) in enumerate(approach_data.items()):
            times = []
            accuracies = []
            for time_step, accuracy in enumerate(experiment_data[:, task_idx]):
                if accuracy != 0:
                    times.append(time_step)
                    accuracies.append(accuracy)
            # ax.plot(times, accuracies, marker='o', label=f'{key}')
            ax.plot(times, accuracies, linestyle=line_styles[exp_idx % len(line_styles)], marker=markers[exp_idx % len(markers)], label=key)
        
        ax.set_title(f'Task {task_idx}')
        ax.set_xlabel('Tasks')
        ax.set_ylabel('Accuracy')
        ax.set_xticks(range(experiment_data.shape[0]))
        ax.grid(True)
        # ax.legend()

    # Hide unused subplots
    for i in range(num_tasks, rows * plots_per_row):
        row, col = divmod(i, plots_per_row)
        axes[row, col].axis('off')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=data_dict['M'])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'plots/{experiment_name}.jpg', format='jpg')
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_plots.py <experiment_name>")
        sys.exit(1)
    generate_plot(sys.argv[1])