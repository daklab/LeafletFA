# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import torch
import sys
from scipy import stats
from scipy.stats import norm

# Increase the font scale for seaborn
sns.set(font_scale=1.5)  # Adjust this value to increase or decrease font size
sns.set_style("white")  # Set the background to white

sys.path.append('/gpfs/commons/home/kisaev/Leaflet-private/src/beta-dirichlet-factor/')
from estimate_bayesian_fdr import *
from differential_splicing import *

# %%
# Main path set up
main_path="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/Simulations/2025/manuscript_sim_analysis/0205"

# List the directories in the main path
dirs = os.listdir(main_path)
# Which dirs have "NoCellType_Random" in them 
dirs = [d for d in dirs if "NoCellType_Random" in d]

# %%
# Choose which simulation directory to analyze 

for i, d in enumerate(dirs):
    print(f"{i}: {d}")
    dir_analyze = dirs[i]
    
    # Set up the path to the simulation directory
    sim_dir=os.path.join(main_path,dir_analyze)
    prop_neg = sim_dir.split("_")[-15]

    # Original PSI_df from simulation
    PSI_df = pd.read_csv(os.path.join(sim_dir, "cell_type_psi_df.csv"))

    # Load learned pis vector 
    pis = pd.read_csv(os.path.join(sim_dir, "pi_df.csv"))
    pis = pis.sort_values(by=['Factor'])
    pis = pis["pi"].values

    # let's load the final_results.csv file 
    albf_scores = pd.read_csv(os.path.join(sim_dir, "ALBF_scores.csv"))
    # rename the first two columns to be sim_psi_1 and sim_psi_2
    albf_scores = albf_scores.rename(columns={"0": "latent_psi_1", "1": "latent_psi_2"})

    print("Number of ALBF values greater than 1: ", len(albf_scores[albf_scores["ALBF"] > 1]))
    print("Number of ALBF values less than 1: ", len(albf_scores[albf_scores["ALBF"] < 1]))
    print("Number of ALBF values less than 0: ", len(albf_scores[albf_scores["ALBF"] < 0]))

    em = EMDifferentialSplicing(albf_scores["ALBF"], initial_p=0.1, max_iter=200)
    results = em.fit()
    albf_scores["posterior_probs"] = results["posterior_probs"]

    # lets get percentile ALBF values of negative labels for calibration analysis of positive labels
    threshold = [0.8, 0.9, 0.95, 0.975, 0.99]

    for thres in threshold:
        # make a new column for each percentile in albf_scores 
        albf_scores[f"FDR_thres_{thres}"] = False
        # check ALBF values if tehy are greater than the percentile value label them as sig 
        albf_scores[f"FDR_thres_{thres}"] = albf_scores["posterior_probs"] >= thres

        # calculate the number of significant values, get FDR 
        tp = albf_scores[(albf_scores["true_label"] == "positive") & (albf_scores[f"FDR_thres_{thres}"])].shape[0]
        fp = albf_scores[(albf_scores["true_label"] == "negative") & (albf_scores[f"FDR_thres_{thres}"])].shape[0]
        tn = albf_scores[(albf_scores["true_label"] == "negative") & (~albf_scores[f"FDR_thres_{thres}"])].shape[0]
        fn = albf_scores[(albf_scores["true_label"] == "positive") & (~albf_scores[f"FDR_thres_{thres}"])].shape[0]

        print(f"At {thres} threshold on FDR, the expected FDR is {1-(thres):.2f}")
        print(f"At {thres} threshold on FDR, number of true positives: {tp}, number of false positives: {fp}")
        print(f"At {thres} threshold on FDR, number of true negatives: {tn}, number of false negatives: {fn}")

        fdr = fp / (fp + tp)
        fnr = fn / (fn + tp)
        print(f"At {thres} threshold on FDR, False Discovery Rate: {fdr:.2f}")
        print(f"At {thres} threshold on FDR, False Negative Rate: {fnr:.2f}")

        print(f"-----------------------------------")
        print(f"-----------------------------------")

    # reindex the albf_scores using junction_id_index so row index matches 
    albf_scores = albf_scores.reset_index(drop=True)
    albf_scores["log_ALBF"] = np.log1p(albf_scores["ALBF"])

    FDR_ALBF = albf_scores[albf_scores["posterior_probs"] >= 0.95].sort_values(by="ALBF").iloc[0]["ALBF"]   
    print(f"The ALBF at the 5% FDR threshold is: {FDR_ALBF}")

    # Save some plots 
    print(f"Saving plots for {sim_dir}")

    # Compare the ALBF values distribution between positive and negative labels 
    plt.figure(figsize=(6, 6))
    sns.violinplot(x="true_label", y="log_ALBF", data=albf_scores)
    plt.savefig(os.path.join(sim_dir, "ALBF_violin_plot.png"))
    plt.close()

    # Plot 
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=albf_scores[albf_scores["true_label"] == "positive"], x="difference", y="ALBF")
    pearson_rho = albf_scores[albf_scores["true_label"] == "positive"]["difference"].corr(albf_scores[albf_scores["true_label"] == "positive"]["ALBF"])
    spearman_rho = albf_scores[albf_scores["true_label"] == "positive"]["difference"].corr(albf_scores[albf_scores["true_label"] == "positive"]["ALBF"], method="spearman")
    plt.title(f"Predefined positive labels: \nPearson's rho = {pearson_rho:.2f}, \nSpearman's rho = {spearman_rho:.2f}")
    print(f"Pearson's rho: {pearson_rho:.2f}, Spearman's rho: {spearman_rho:.2f} for comparing ALBF and simulated difference")
    plt.savefig(os.path.join(sim_dir, "ALBF_vs_simulated_difference.png"))
    plt.close()

    sns.scatterplot(data=albf_scores[albf_scores["true_label"] == "positive"], x="delta_est", y="ALBF")
    pearson_rho = albf_scores[albf_scores["true_label"] == "positive"]["delta_est"].corr(albf_scores[albf_scores["true_label"] == "positive"]["ALBF"])
    spearman_rho = albf_scores[albf_scores["true_label"] == "positive"]["delta_est"].corr(albf_scores[albf_scores["true_label"] == "positive"]["ALBF"], method="spearman")
    plt.title(f"Predefined positive labels: \nPearson's rho = {pearson_rho:.2f}, \nSpearman's rho = {spearman_rho:.2f}")
    print(f"Pearson's rho: {pearson_rho:.2f}, Spearman's rho: {spearman_rho:.2f} for comparing ALBF and estimated difference")
    plt.savefig(os.path.join(sim_dir, "ALBF_vs_estimated_difference.png"))
    plt.close()

    sns.scatterplot(data=albf_scores[albf_scores["true_label"] == "positive"], x="delta_est", y="difference")
    pearson_rho = albf_scores[albf_scores["true_label"] == "positive"]["delta_est"].corr(albf_scores[albf_scores["true_label"] == "positive"]["difference"])
    spearman_rho = albf_scores[albf_scores["true_label"] == "positive"]["delta_est"].corr(albf_scores[albf_scores["true_label"] == "positive"]["difference"], method="spearman")
    plt.title(f"Predefined positive labels: \nPearson's rho = {pearson_rho:.2f}, \nSpearman's rho = {spearman_rho:.2f}")
    print(f"Pearson's rho: {pearson_rho:.2f}, Spearman's rho: {spearman_rho:.2f} for comparing simulated and estimated difference")
    plt.savefig(os.path.join(sim_dir, "simulated_vs_estimated_difference.png"))
    plt.close()

    # lets get percentile ALBF values of negative labels for calibration analysis of positive labels
    percentiles = albf_scores[albf_scores["true_label"] == "negative"]["ALBF"].describe(percentiles=[0.5, 0.6, 0.7, 0.8, 0.85, 0.95, 0.975, 0.99])
    percentiles = percentiles[["50%", "60%", "70%", "80%", "85%", "95%", "97.5%", "99%"]]

    for perc, value in percentiles.items():
        # make a new column for each percentile in albf_scores 
        albf_scores[f"percentile_{perc}"] = False
        # check ALBF values if tehy are greater than the percentile value label them as sig 
        albf_scores[f"significant_{perc}"] = albf_scores["ALBF"] >= value
        print(f"The ALBF value at {perc} percentile is: {value}")

        # calculate the number of significant values, get FDR 
        tp = albf_scores[(albf_scores["true_label"] == "positive") & (albf_scores[f"significant_{perc}"])].shape[0]
        fp = albf_scores[(albf_scores["true_label"] == "negative") & (albf_scores[f"significant_{perc}"])].shape[0]

        perc_num = float(perc[:-1]) / 100
        print(f"At {perc} percentile of null distribution, the expected FDR is {1-(perc_num):.2f}")
        print(f"At {perc} percentile of null distribution, number of positives: {tp}, number of negatives: {fp}")

        fdr = fp / (fp + tp)
        print(f"At {perc} percentile of null distribution, False Discovery Rate: {fdr:.2f}")
        print(f"-----------------------------------")
        print(f"-----------------------------------")

    # save to file the new albf_scores file 
    albf_scores["prop_neg"] = prop_neg
    albf_scores.to_csv(os.path.join(sim_dir, "ALBF_scores_with_FDR.csv"), index=False)
    print(f"Saved the new ALBF_scores_with_FDR.csv file to {sim_dir}")
