import numpy as np

class DifferentialSplicingAnalyzer:
    def __init__(self, mu, v):
        """
        Initialize with matrices of mu (mean) and v (variance) for the logit-normal distributions.
        
        Parameters:
            mu (np.array): A KxJ matrix of means, where K is the number of cell states and J is the number of junctions.
            v (np.array): A KxJ matrix of variances, corresponding to mu.
        
        Example:
            mu = np.random.normal(size=(10, 100))  # 10 cell states, 100 junctions
            v = np.random.uniform(0.1, 1, size=(10, 100))
            analyzer = DifferentialSplicingAnalyzer(mu, v)
        """
        self.mu = mu
        self.v = v

    def calculate_p_h0(self, j):
        """
        Calculate the probability of the data under the null hypothesis for junction j,
        assuming a common mean and variance across all cell states (H0).

        Under H0, we hypothesize that all cell states for a given junction have the same
        splicing efficiency, which simplifies our model to a single Gaussian distribution
        across all states. The formula used here integrates the product of Gaussian PDFs
        across cell states to compute a combined Gaussian PDF, representing the total 
        probability of observing the data under H0.

        Specifically, the method calculates:
            P_j(H_0) = 1 / N(0, mu, v) * prod_k N(0, mu_k, v_k)
        where N(0, mu, v) is the standard normal distribution PDF evaluated at the combined mean
        and variance calculated from all cell states. This is simplified further as:
            N(y; mu, v) is the Gaussian PDF for a combined distribution.
            mu and v are derived by combining individual mu_k and v_k values across cell states.

        Parameters:
            j (int): Index of the junction to analyze.

        Returns:
            float: The probability P(H_0) for the given junction, representing the likelihood
                   of observing the splicing data under the null hypothesis.

        Example:
            p_h0 = analyzer.calculate_p_h0(0)  # Calculate P(H_0) for the first junction
        """
        v_combined = 1 / np.sum(1 / self.v[:, j])  # Combined variance from all states for junction j
        mu_combined = v_combined * np.sum(self.mu[:, j] / self.v[:, j])  # Combined mean using weights

        # The standard normal PDF N(0, mu, v) simplified when evaluated at mu.
        # It provides the scale of deviation from the mean in terms of the combined variance.
        p_h0 = 1 / np.sqrt(2 * np.pi * v_combined) * np.exp(-mu_combined**2 / (2 * v_combined))
        return p_h0

    def calculate_albf(self, j):
        """
        Calculate the Approximate Log Bayes Factor (ALBF) for junction j.
        
        Parameters:
            j (int): index of the junction
            
        Returns:
            float: ALBF for the given junction
        
        Example:
            albf_j1 = analyzer.calculate_albf(0)  # Get ALBF for the first junction
            print(f"ALBF for junction 1: {albf_j1}")
        """
        p_h0 = self.calculate_p_h0(j)
        albf = -np.log(p_h0)
        return albf

    def albf_all_vs_1(self, j, factor_index):
        """
        Calculate ALBF for comparing one factor against all others combined.
        
        Parameters:
            j (int): index of the junction
            factor_index (int): index of the factor to compare against all others
        
        Returns:
            float: ALBF for the given junction and factor
        
        Example:
            albf_factor1_vs_others = analyzer.albf_all_vs_1(0, 0)
            print(f"ALBF for factor 1 vs others at junction 1: {albf_factor1_vs_others}")
        """
        other_indices = [k for k in range(self.mu.shape[0]) if k != factor_index]

        # Combined parameters for all but the selected factor
        v_other_combined = 1 / np.sum(1 / self.v[other_indices, j])
        mu_other_combined = v_other_combined * np.sum(self.mu[other_indices, j] / self.v[other_indices, j])

        # Parameters for the selected factor
        mu_1 = self.mu[factor_index, j]
        v_1 = self.v[factor_index, j]

        # Calculate P(H_0) using the two groups: selected factor vs. all others
        mu_combined = v_other_combined * mu_other_combined + v_1 * mu_1 / (v_other_combined + v_1)
        v_combined = 1 / (1 / v_other_combined + 1 / v_1)

        p_h0 = 1 / np.sqrt(2 * np.pi * v_combined) * np.exp(-mu_combined**2 / (2 * v_combined))
        albf = -np.log(p_h0)

        return albf
