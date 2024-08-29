import numpy as np
import torch 

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
        assert torch.is_tensor(mu), "mu must be a PyTorch tensor"
        assert torch.is_tensor(v), "v must be a PyTorch tensor"
        self.mu = mu
        self.v = v

    def ensure_positive_variance(self, variance):
        min_variance = 1e-6  # Minimum variance threshold to avoid division by zero
        return torch.clamp(variance, min=min_variance)

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
        v_combined = 1 / torch.sum(1 / self.v[:, j])
        # Ensure that the combined variance is positive
        v_combined = self.ensure_positive_variance(v_combined)
        mu_combined = v_combined * torch.sum(self.mu[:, j] / self.v[:, j])
        
        # Avoid large values for mu_combined squared over v_combined
        exponent = -mu_combined.pow(2) / (2 * v_combined)
        exponent = torch.clamp(exponent, max=0)  # Limit the exponent to zero or less to prevent overflow

        gaussian_at_zero_combined = 1 / torch.sqrt(2 * torch.pi * v_combined) * torch.exp(exponent)


        # Evaluate the Gaussian PDF at zero for each cell state with clamping
        exponent_individual = -self.mu[:, j].pow(2) / (2 * self.v[:, j])
        exponent_individual = torch.clamp(exponent_individual, max=0)
        gaussians_at_zero_individual = 1 / torch.sqrt(2 * torch.pi * self.v[:, j]) * torch.exp(exponent_individual)
        
        # Product of individual Gaussians evaluated at zero
        product_gaussians = torch.prod(gaussians_at_zero_individual)
        
        # Prevent division by zero or very small values in gaussian_at_zero_combined
        gaussian_at_zero_combined = max(gaussian_at_zero_combined, 1e-10)

        # Normalize this product by the Gaussian PDF of the combined parameters evaluated at zero
        p_h0 = product_gaussians / gaussian_at_zero_combined
        
        return p_h0

    def combined_mean_variance(self, means, variances):
        inv_variances = 1 / variances
        combined_variance = 1 / torch.sum(inv_variances)
        combined_mean = combined_variance * torch.sum(means * inv_variances)
        return combined_mean, combined_variance

    def gaussian_pdf(self, x, mean, std):
        var = std ** 2
        denom = (2 * torch.pi * var) ** 0.5
        num = torch.exp(- (x - mean) ** 2 / (2 * var))
        return num / denom
    
    def likelihood_under_null(self, observed_data, combined_mean, combined_std, means, stds):
        combined_pdf = self.gaussian_pdf(0, combined_mean, combined_std)

        likelihood = combined_pdf
        for mean, std in zip(means, stds):
            likelihood *= self.gaussian_pdf(0, mean, std)

        for data in observed_data:
            likelihood *= self.gaussian_pdf(data, combined_mean, combined_std)

        return likelihood
        
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
        # Use a lower bound to avoid negative infinity in log
        epsilon = 1e-10
        p_h0 = torch.clamp(p_h0, min=epsilon)
        albf = -torch.log(p_h0)

        return albf

    def albf_all_vs_1(self, j, factor_index):
        # Ensure factor_index is within bounds
        assert 0 <= factor_index < self.mu.shape[0], "factor_index out of bounds"
    
        other_indices = [k for k in range(self.mu.shape[0]) if k != factor_index]
    
        # Combined parameters for all but the selected factor
        v_other_combined = 1 / torch.sum(1 / self.v[other_indices, j])
        mu_other_combined = v_other_combined * torch.sum(self.mu[other_indices, j] / self.v[other_indices, j])
    
        # Parameters for the selected factor
        mu_1 = self.mu[factor_index, j]
        v_1 = self.v[factor_index, j]
    
        # Calculate combined mu and v for the comparison
        mu_combined = v_other_combined * mu_other_combined + v_1 * mu_1 / (v_other_combined + v_1)
        v_combined = 1 / (1 / v_other_combined + 1 / v_1)
    
        # Ensure variance is not zero
        v_combined = self.ensure_positive_variance(v_combined)
    
        # Calculate the Gaussian PDF and ALBF
        p_h0 = 1 / torch.sqrt(2 * torch.pi * v_combined) * torch.exp(-mu_combined**2 / (2 * v_combined))
        albf = -torch.log(p_h0)
    
        return albf
