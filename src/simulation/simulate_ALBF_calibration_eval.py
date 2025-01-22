import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class ALBFCalibrationAnalysis:
    """
    A class to analyze and validate ALBF (Approximate Log Bayes Factor) calibration
    """
    def __init__(self, albf_values, true_labels, num_bins=20):
        self.albf_values = np.array(albf_values)
        self.true_labels = np.array(true_labels)
        self.num_bins = num_bins
        
    def compute_null_distribution(self):
        """Compute empirical null distribution from non-DS junctions"""
        null_albfs = self.albf_values[self.true_labels == 0]
        # Fit normal distribution to null ALBFs
        mu, std = stats.norm.fit(null_albfs)
        return mu, std
        
    def normalize_albf(self):
        """Normalize ALBF values based on null distribution"""
        mu, std = self.compute_null_distribution()
        return (self.albf_values - mu) / std
        
    def compute_calibration_curve(self):
        """Compute calibration curve showing empirical vs expected FDR"""
        sorted_idx = np.argsort(-self.albf_values)  # Sort in descending order
        sorted_labels = self.true_labels[sorted_idx]
        
        # Calculate empirical FDR at different cutoffs
        thresholds = np.linspace(0, 1, self.num_bins)
        empirical_fdr = []
        expected_fdr = []
        
        for thresh in thresholds:
            cutoff_idx = int(len(sorted_labels) * thresh)
            if cutoff_idx > 0:
                # Empirical FDR = proportion of false positives
                fdr = 1 - np.mean(sorted_labels[:cutoff_idx])
                empirical_fdr.append(fdr)
                expected_fdr.append(thresh)
                
        return np.array(expected_fdr), np.array(empirical_fdr)
    
    def compute_calibration_metrics(self):
        """Compute metrics to assess calibration quality"""
        expected_fdr, empirical_fdr = self.compute_calibration_curve()
        
        # Calculate calibration error
        calibration_error = np.mean(np.abs(empirical_fdr - expected_fdr))
        
        # Calculate calibration slope and R²
        slope, intercept, r_value, _, _ = stats.linregress(expected_fdr, empirical_fdr)
        r_squared = r_value ** 2
        
        return {
            'calibration_error': calibration_error,
            'calibration_slope': slope,
            'r_squared': r_squared
        }
    
    def plot_calibration_diagnostics(self, output_dir):
        """Generate diagnostic plots for ALBF calibration"""
        # 1. Plot calibration curve
        expected_fdr, empirical_fdr = self.compute_calibration_curve()
        metrics = self.compute_calibration_metrics()
        
        plt.figure(figsize=(6, 6))
        plt.plot(expected_fdr, empirical_fdr, 'b-', label='Observed')
        plt.plot([0, 1], [0, 1], 'k--', label='Ideal')
        plt.xlabel('Expected FDR')
        plt.ylabel('Empirical FDR')
        plt.title(f'ALBF Calibration Curve\nCalibration Error: {metrics["calibration_error"]:.3f}')
        plt.legend()
        plt.savefig(f'{output_dir}/albf_calibration_curve.pdf', bbox_inches='tight')
        plt.close()
        
        # 2. Plot normalized ALBF distributions
        normalized_albf = self.normalize_albf()
        plt.figure(figsize=(6, 6))
        sns.kdeplot(data=normalized_albf[self.true_labels == 0], 
                   label='Non-DS (Null)', color='blue')
        sns.kdeplot(data=normalized_albf[self.true_labels == 1], 
                   label='DS (Alternative)', color='red')
        plt.xlabel('Normalized ALBF')
        plt.ylabel('Density')
        plt.title('Distribution of Normalized ALBF Values')
        plt.legend()
        plt.savefig(f'{output_dir}/normalized_albf_distribution.pdf', bbox_inches='tight')
        plt.close()
        
        return metrics

def validate_albf_thresholds(albf_values, true_labels, fdr_targets=[0.01, 0.05, 0.1]):
    """
    Validate ALBF thresholds at different target FDR levels
    
    Parameters:
    -----------
    albf_values : array-like
        ALBF values for each junction
    true_labels : array-like
        True differential splicing status (1 for DS, 0 for non-DS)
    fdr_targets : list
        Target FDR levels to evaluate
        
    Returns:
    --------
    dict : Dictionary containing validation results for each target FDR
    """
    results = {}
    sorted_idx = np.argsort(-albf_values)
    sorted_albf = albf_values[sorted_idx]
    sorted_labels = true_labels[sorted_idx]
    
    for target_fdr in fdr_targets:
        # Find threshold that achieves target FDR
        for i in range(len(sorted_albf)):
            empirical_fdr = 1 - np.mean(sorted_labels[:i+1])
            if empirical_fdr > target_fdr:
                break
                
        threshold = sorted_albf[i]
        num_discoveries = i + 1
        true_positives = np.sum(sorted_labels[:i+1])
        
        results[target_fdr] = {
            'threshold': threshold,
            'num_discoveries': num_discoveries,
            'true_positives': true_positives,
            'empirical_fdr': empirical_fdr,
            'power': true_positives / np.sum(true_labels)
        }
        
    return results