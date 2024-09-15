# Fully reproducible code with all necessary imports and setup

# Import necessary packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Set seed for reproducibility
np.random.seed(42)

# Simulate data for 3 cell states
J = 200  # Number of junctions
psis = np.random.beta(0.5, 0.5, size=(3, J))  # Simulate psis for 3 cell states

# Picking a specific junction for demonstration
jj = 163

# Define a new concentration parameter for clarity
bb_conc = 50

# Flag for plotting
plot_bb = True

# Start plotting with modifications
if plot_bb:
    # Generate values for the x-axis
    x = np.linspace(0, 1, 1000)

    # Success probabilities for the selected junction across 3 cell states
    success_probs = psis[:, jj]
    conc_params = [10, bb_conc, 1000]

    # Create a new figure with larger font sizes and clearer titles
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Increase font size globally for better readability
    plt.rcParams.update({'font.size': 14})

    # Define colors for each state
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    # Plot individual Beta distributions for different concentration parameters
    for i, conc_param in enumerate(conc_params):
        for j, (success_prob, color) in enumerate(zip(success_probs, colors)):
            bb_alpha = success_prob * conc_param
            bb_beta = (1 - success_prob) * conc_param

            # Calculate the PDF of the beta distribution
            pdf = beta.pdf(x, bb_alpha, bb_beta)

            # Plot the PDF with distinct colors
            axs[i].plot(x, pdf, label=f'State {j+1}', color=color)

        # Set titles for each plot
        axs[i].set_title(f'Cell State-Specific Beta Distributions\n(Concentration: {conc_param})', fontsize=16)
        axs[i].set_xlabel('Probability of Junction Usage', fontsize=14)
        axs[i].set_ylabel('Density', fontsize=14)
        axs[i].grid(True)

    # Remove legends for clarity
    for ax in axs:
        ax.legend().remove()

    plt.tight_layout()
    plt.show()
