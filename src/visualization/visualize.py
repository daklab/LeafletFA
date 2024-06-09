# TO-DO

#1. plot coverage across transcripts from BAM files --> bigwig deeptools, compare R vs T reads 
#2. plot number of UMIs per gene per cell 

J = psis.shape[1]
jj = np.random.randint(J)
jj = 163
from scipy.stats import beta

if plot_bb:

    # Generate values for the x-axis
    x = np.linspace(0, 1, 1000)

    success_probs = psis[:,jj]
    bbconc_param = bb_conc
    print(f"The bb conc param is {bbconc_param}")
    conc_params = [10, bbconc_param, 1000]
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))
    # Define a single legend for all curves
    legend_labels = []

    for i, conc_param in enumerate(conc_params):

        # Plot individual curves for each success probability
        for j, success_prob in enumerate(success_probs):
            # Calculate shape parameters for the Beta distribution
            bb_alpha = success_prob * conc_param
            bb_beta = (1 - success_prob) * conc_param

            # Calculate the probability density function (PDF) of the beta distribution
            pdf = beta.pdf(x, bb_alpha, bb_beta)

            # Plot the Beta distribution
            # axs[i].plot(x, pdf, label=f'Success Prob {j+1}: {success_prob:.2f}')
            # Plot the Beta distribution and store the label for the legend
            line, = axs[i].plot(x, pdf, label=f'Success Prob {jj+1}: {success_prob:.2f}')
            if i == 0:  # Add legend labels only once
                legend_labels.append(line.get_label())
            
        axs[i].set_title(f'Beta Distribution (Cell State: {conc_param})', fontsize=10)
        axs[i].set_xlabel('x')
        axs[i].set_ylabel('Probability Density')
        axs[i].grid(True)

    # Also plot the Beta distribution average junction behaviour using a_j and b_j values
    # in this case just a[1] and b[1] for the current junction 
    bb_alpha = a[jj]
    bb_beta = b[jj]
    bb_avg = bb_alpha / (bb_alpha+bb_beta)
    print(f"The average junction behaviour is: {bb_avg:.2f}")
    pdf = beta.pdf(x, bb_alpha, bb_beta)
    axs[3].plot(x, pdf, label=f'Average Junction Behaviour')
    axs[3].set_title('Beta Distribution (Average Junction Behaviour)', fontsize=10)
    axs[3].set_xlabel('x')
    axs[3].set_ylabel('Probability Density')
    axs[3].grid(True)
    
    # extract only unique values from legend lines 
    fig.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    # print the plot
    # print axs[2]

    plt.show()