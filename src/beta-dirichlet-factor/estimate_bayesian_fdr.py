import numpy as np
from scipy.special import expit

class EMDifferentialSplicing:
    def __init__(self, albf_scores, max_iter=100, tolerance=1e-4, initial_p=0.5):
        """
        Initialize EM algorithm with log-transform of ALBF scores.
        """
        self.albf_scores = np.array(albf_scores)        

        # Center ALBF first then logit 
        self.b_j = np.clip(expit(self.albf_scores - np.median(self.albf_scores)), 1e-10, 1-1e-10)  # Centering before sigmoid

        self.max_iter = max_iter
        self.tolerance = tolerance
        self.p = initial_p
        self.n_junctions = len(albf_scores)
        
        # Print initial statistics
        print("Initial Statistics:")
        print(f"Original ALBF range: [{np.min(albf_scores):.2f}, {np.max(albf_scores):.2f}]")
        print(f"b_j range: [{np.min(self.b_j):.4f}, {np.max(self.b_j):.4f}]")
        
    def e_step(self):
        """
        E-step: Compute q(s_j=1) for each junction.
        """
        # Keep p near 0.5
        p_safe = np.clip(self.p, 0.1, 0.9)
        
        # Compute log odds
        log_odds_b = np.log(np.clip(self.b_j, 1e-10, 1-1e-10)) - np.log(1 - np.clip(self.b_j, 1e-10, 1-1e-10))
        log_odds_p = np.log(p_safe) - np.log(1 - p_safe)

        # Compute posterior probabilities
        total_log_odds = log_odds_b + log_odds_p
        q_s1 = np.clip(expit(total_log_odds), 1e-10, 1-1e-10)
        
        return q_s1
    
    def m_step(self, q_s1):
        """
        M-step: Update the prior probability p with tight bounds around 0.5.
        """
        return np.clip(np.mean(q_s1), 0.1, 0.9)
    
    def compute_log_likelihood(self, q_s1):
        """
        Compute log likelihood.
        """
        # s_j in likelihood is 1 if junction is differentially spliced or 0 otherwise
        # should sample it using q_s1 and bernoulli distribution
        return np.sum(q_s1 * np.log(self.b_j) + (1 - q_s1) * np.log(1 - self.b_j))
    
    def fit(self):
        """
        Run the EM algorithm until convergence or max iterations.
        """
        prior_trajectory = [self.p]
        prev_log_likelihood = -np.inf
        best_likelihood = -np.inf
        best_params = None
        
        for iteration in range(self.max_iter):
            # E-step
            q_s1 = self.e_step()
            
            # M-step
            new_p = self.m_step(q_s1)
            self.p = new_p
            
            # Compute log likelihood
            log_likelihood = self.compute_log_likelihood(q_s1)
            
            # Store best parameters
            if log_likelihood > best_likelihood:
                best_likelihood = log_likelihood
                best_params = {
                    'posterior_probs': q_s1.copy(),
                    'prior_prob': new_p,
                    'log_likelihood': log_likelihood
                }
            
            # Print iteration statistics
            print(f"\nIteration {iteration + 1}:")
            print(f"p: {self.p:.6f}")
            print(f"log likelihood: {log_likelihood:.6f}")
            print(f"q_s1 range: [{np.min(q_s1):.6f}, {np.max(q_s1):.6f}]")
            
            # Check for convergence
            if iteration > 0:
                improvement = log_likelihood - prev_log_likelihood
                if improvement < -1e-10:
                    print("\nLikelihood decreased, using best parameters")
                    return {**best_params, 'converged': True, 'n_iterations': iteration}
                elif improvement < self.tolerance:
                    print("\nConverged!")
                    return {**best_params, 'converged': True, 'n_iterations': iteration}
            
            prior_trajectory.append(self.p)
            prev_log_likelihood = log_likelihood
        
        return {**best_params, 'converged': False, 'n_iterations': self.max_iter}