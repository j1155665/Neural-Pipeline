"""
Accumulated Evidence Analysis - Compare MST vs VPS temporal integration weights
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')


class AccumulatedEvidenceAnalyzer:
    """
    Analyze accumulated decision evidence from MST and VPS decoders.
    Tests feedforward vs feedback by comparing temporal integration weights.
    Main methods: load data, integrate DVs, fit regression, bootstrap CIs.
    """
    
    def __init__(self, subject, dates, regress_type='noregress', results_dir=None, save_dir=None):
        """
        Initialize analyzer with subject info and directories.
        Sets up paths for loading decoder results and saving outputs.
        Creates necessary output directories if they don't exist.
        """
        self.subject = subject
        self.dates = dates if isinstance(dates, list) else [dates]
        self.regress_type = regress_type
        
        if results_dir is None:
            self.results_dir = Path(rf'D:\Neural-Pipeline\results\analysis_pseudopopulation\decoders_{regress_type}')
        else:
            self.results_dir = Path(results_dir)
        
        if save_dir is None:
            self.save_dir = Path(rf'D:\Neural-Pipeline\results\accumulated_evidence')
        else:
            self.save_dir = Path(save_dir)
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / 'processed_data').mkdir(exist_ok=True)
        (self.save_dir / 'model_fits').mkdir(exist_ok=True)
        (self.save_dir / 'figures').mkdir(exist_ok=True)
    
    def load_decoder_results(self, date, area, target, alignment, mod, coh):
        """
        Load a single decoder result file.
        Returns None if file not found.
        File format: subject_date_area_target_alignment_train_modX_cohX_test_modX_cohX_results.npy
        """
        filename = (f"{self.subject}_{date}_{area}_{target}_{alignment}_"
                   f"train_mod{mod}_coh{coh}_test_mod{mod}_coh{coh}_results.npy")
        filepath = self.results_dir / filename
        
        if filepath.exists():
            return np.load(filepath, allow_pickle=True).item()
        else:
            return None
    
    def extract_dv_timeseries(self, results, alignment, test_sets=None, target='choice'):
        """
        Extract DV timeseries using predicted probabilities directly.
        Returns: dv_matrix (n_trials × n_timepoints), behavior dict, time_axis.
        
        Uses y_proba (probability of class 1) as the decision variable.
        No sign flipping needed - decoder outputs are already correctly oriented.
        
        DV interpretation:
            - Values near 1.0: decoder confidently predicts class 1
            - Values near 0.0: decoder confidently predicts class 0
            - Values near 0.5: decoder is uncertain
        """
        if results is None:
            return None, None, None
        
        if test_sets is None:
            test_sets = ['test_cv']  # Default to just CV, not all test sets
        
        trial_results = results['trial_results']
        time_axes = results.get('time_axes', {})
        time_axis = time_axes.get(alignment, None)
        
        dv_by_time = {}
        behavior_by_time = {}
        
        for r in trial_results:
            t = r['time']
            if t not in dv_by_time:
                dv_by_time[t] = []
                behavior_by_time[t] = {}
            
            for test_set in test_sets:
                if test_set not in r:
                    continue
                
                test_data = r[test_set]
                y_proba = test_data.get('y_proba', np.array([]))
                behavior = test_data.get('behavior', {})
                
                if len(y_proba) == 0 or len(behavior) == 0:
                    continue
                
                # Use y_proba directly as decision variable
                # y_proba = P(class 1 | neural activity at time t)
                # Higher values → stronger evidence for class 1
                # Lower values → stronger evidence for class 0
                dv = y_proba.copy()
                
                # Append DVs
                dv_by_time[t].extend(dv.tolist())
                
                # Collect behavior
                for key, value in behavior.items():
                    if key not in behavior_by_time[t]:
                        behavior_by_time[t][key] = []
                    behavior_by_time[t][key].append(value)
        
        if len(dv_by_time) == 0:
            return None, None, None
        
        # Convert to matrix
        time_points = sorted(dv_by_time.keys())
        n_trials = len(dv_by_time[time_points[0]])
        n_times = len(time_points)
        
        dv_matrix = np.zeros((n_trials, n_times))
        for i, t in enumerate(time_points):
            dv_matrix[:, i] = np.array(dv_by_time[t])
        
        # Concatenate behavior
        behavior = {}
        for key, value_list in behavior_by_time[time_points[0]].items():
            try:
                behavior[key] = np.concatenate(value_list)
            except:
                behavior[key] = np.array([])
        
        # Convert time indices to actual times
        if time_axis is not None:
            time_values = np.array([time_axis[t] for t in time_points])
        else:
            time_values = np.array(time_points)
        
        return dv_matrix, behavior, time_values
    
    def load_all_sessions(self, area, target, alignment, conditions=None, dates=None, test_sets=None):
        """
        Load and concatenate data across sessions and conditions.
        Now includes target parameter for proper DV flipping.
        """
        if conditions is None:
            conditions = [(1, 1), (2, 1), (2, 2), (3, 1), (3, 2)]
        
        if dates is None:
            dates = self.dates
        elif isinstance(dates, str):
            dates = [dates]
        
        if test_sets is None:
            test_sets = ['test_cv']  # Default to just CV
        
        all_dv = []
        all_choice = []
        all_pdw = []
        all_rt = []
        all_modality = []
        all_coherence = []
        all_heading = []
        time_axis = None
        
        for date in dates:
            for mod, coh in conditions:
                if mod == 1 and coh == 2:
                    continue
                
                results = self.load_decoder_results(date, area, target, alignment, mod, coh)
                if results is None:
                    continue
                
                # Extract with proper target for DV flipping
                dv_matrix, behavior, time_values = self.extract_dv_timeseries(
                    results, alignment, test_sets=test_sets, target=target
                )
                
                if dv_matrix is None:
                    continue
                
                if time_axis is None:
                    time_axis = time_values
                
                choice = behavior.get('choice', np.array([]))
                choice = (choice - 1).astype(int) if len(choice) > 0 and np.max(choice) > 1 else choice
                
                pdw = behavior.get('PDW', np.array([]))
                rt = behavior.get('RT', np.array([]))
                heading = behavior.get('heading', np.array([]))
                
                n_trials = len(choice)
                
                all_dv.append(dv_matrix)
                all_choice.append(choice)
                all_pdw.append(pdw)
                all_rt.append(rt)
                all_modality.append(np.full(n_trials, mod))
                all_coherence.append(np.full(n_trials, coh))
                all_heading.append(heading)
        
        if len(all_dv) == 0:
            return None
        
        return {
            'dv_matrix': np.vstack(all_dv),
            'choice': np.concatenate(all_choice),
            'PDW': np.concatenate(all_pdw),
            'RT': np.concatenate(all_rt),
            'modality': np.concatenate(all_modality),
            'coherence': np.concatenate(all_coherence),
            'heading': np.concatenate(all_heading),
            'time_axis': time_axis,
            'n_trials': sum(len(c) for c in all_choice),
            'area': area,
            'target': target,
            'alignment': alignment
        }
    
    def match_trials_across_areas(self, data_mst, data_vps):
        """
        Match trials between MST and VPS (ensure same trials).
        Takes minimum number of trials if counts differ.
        Returns dict with matched dv_mst, dv_vps, and behavior.
        """
        n_mst = data_mst['n_trials']
        n_vps = data_vps['n_trials']
        n_min = min(n_mst, n_vps)
        
        return {
            'dv_mst': data_mst['dv_matrix'][:n_min],
            'dv_vps': data_vps['dv_matrix'][:n_min],
            'choice': data_mst['choice'][:n_min],
            'PDW': data_mst['PDW'][:n_min],
            'RT': data_mst['RT'][:n_min],
            'modality': data_mst['modality'][:n_min],
            'coherence': data_mst['coherence'][:n_min],
            'heading': data_mst['heading'][:n_min],
            'time_axis': data_mst['time_axis'],
            'n_trials': n_min
        }
    
    def define_time_window(self, time_axis, window_start, window_end):
        """
        Find time indices within specified window.
        Returns boolean mask for time points in [window_start, window_end].
        Example: (0.0, 0.5) for 0-500ms post-stimulus.
        """
        mask = (time_axis >= window_start) & (time_axis <= window_end)
        return mask
    
    def integrate_dv_uniform(self, dv_matrix, time_indices):
        """
        Integrate DV with uniform weights (simple sum).
        Sums DV across specified time window for each trial.
        Returns: integrated_dv array (n_trials,).
        """
        if isinstance(time_indices, np.ndarray) and time_indices.dtype == bool:
            dv_windowed = dv_matrix[:, time_indices]
        else:
            dv_windowed = dv_matrix[:, time_indices]
        
        return np.sum(dv_windowed, axis=1)
    
    def integrate_dv_exponential(self, dv_matrix, time_axis, time_indices, lambda_param):
        """
        Integrate DV with exponential weights: w(t) = exp(λ*t).
        λ > 0: recency (late evidence weighted more), λ < 0: primacy (early more).
        Returns: weighted sum for each trial.
        """
        if isinstance(time_indices, np.ndarray) and time_indices.dtype == bool:
            dv_windowed = dv_matrix[:, time_indices]
            time_windowed = time_axis[time_indices]
        else:
            dv_windowed = dv_matrix[:, time_indices]
            time_windowed = time_axis[time_indices]
        
        weights = np.exp(lambda_param * time_windowed)
        integrated_dv = np.sum(dv_windowed * weights[np.newaxis, :], axis=1)
        
        return integrated_dv
    
    def integrate_dv_gaussian(self, dv_matrix, time_axis, time_indices, center, sigma):
        """
        Integrate DV with Gaussian weights: w(t) = exp(-0.5 * ((t - center) / sigma)^2).
        center: peak of the Gaussian (time point with highest weight)
        sigma: width of the Gaussian (standard deviation)
        Returns: weighted sum for each trial.
        """
        if isinstance(time_indices, np.ndarray) and time_indices.dtype == bool:
            dv_windowed = dv_matrix[:, time_indices]
            time_windowed = time_axis[time_indices]
        else:
            dv_windowed = dv_matrix[:, time_indices]
            time_windowed = time_axis[time_indices]
        
        # Compute Gaussian weights
        weights = np.exp(-0.5 * ((time_windowed - center) / sigma) ** 2)
        weights = weights / np.sum(weights)  # Normalize to sum to 1
        
        integrated_dv = np.sum(dv_windowed * weights[np.newaxis, :], axis=1)
        
        return integrated_dv
    
    def select_optimal_sigma_for_area(self, dv_matrix, time_axis, time_indices, choices,
                                  sigma_values=None, n_centers=4):
        """
        Select optimal sigma for ONE area by testing across multiple center positions.
        Uses MAXIMUM AUC across all centers (finds sigma that performs best at its peak).
        
        Strategy: Test each sigma at multiple time points. The sigma that achieves the
        highest peak AUC (regardless of where) is considered optimal.
        
        Parameters:
        -----------
        dv_matrix : ndarray (n_trials, n_timepoints)
            Decision variable matrix for ONE area (MST or VPS)
        time_axis : ndarray
            Time values for each timepoint
        time_indices : bool array or int array
            Which timepoints to include in integration window
        choices : ndarray
            Binary choice labels (0 or 1)
        sigma_values : list/array
            Sigma values to test [default: np.linspace(0.01, 0.3, 20)]
        n_centers : int
            Number of center positions to test (at 25%, 50%, 75%, 100% of window)
        
        Returns:
        --------
        dict with:
            - 'optimal_sigma': best sigma value (highest max AUC)
            - 'max_auc_by_sigma': maximum AUC achieved by each sigma
            - 'mean_auc_by_sigma': mean AUC for each sigma
            - 'auc_grid': (n_centers × n_sigmas) array with all AUCs
            - 'center_positions': array of tested center values (in seconds)
            - 'sigma_values': array of tested sigma values
            - 'best_center_by_sigma': which center gave max AUC for each sigma
        """
        # Extract time window
        if isinstance(time_indices, np.ndarray) and time_indices.dtype == bool:
            time_windowed = time_axis[time_indices]
        else:
            time_windowed = time_axis[time_indices]
        
        window_start = time_windowed.min()
        window_end = time_windowed.max()
        window_duration = window_end - window_start
        
        # Define center positions (25%, 50%, 75%, 100%)
        percentiles = np.linspace(0.25, 1.0, n_centers)
        center_positions = window_start + percentiles * window_duration
        
        # Default sigma values - changed to 0.01 to 0.3
        if sigma_values is None:
            sigma_values = np.linspace(0.01, 0.3, 20)
        else:
            sigma_values = np.array(sigma_values)
        
        # Initialize AUC grid
        auc_grid = np.zeros((n_centers, len(sigma_values)))
        
        # Grid search
        for i, center in enumerate(center_positions):
            for j, sigma in enumerate(sigma_values):
                # Integrate DV with this Gaussian
                integrated_dv = self.integrate_dv_gaussian(
                    dv_matrix, time_axis, time_indices, center, sigma
                )
                
                # Calculate AUC
                try:
                    auc = roc_auc_score(choices, integrated_dv)
                    # Handle flipped decoders
                    if auc < 0.5:
                        auc = 1 - auc
                    auc_grid[i, j] = auc
                except:
                    auc_grid[i, j] = 0.5
        
        # Calculate MAX AUC for each sigma (new approach)
        max_auc_by_sigma = np.max(auc_grid, axis=0)
        mean_auc_by_sigma = np.mean(auc_grid, axis=0)
        best_center_idx_by_sigma = np.argmax(auc_grid, axis=0)
        best_center_by_sigma = center_positions[best_center_idx_by_sigma]
        
        # Select optimal sigma (highest max AUC)
        optimal_idx = np.argmax(max_auc_by_sigma)
        optimal_sigma = sigma_values[optimal_idx]
        
        return {
            'optimal_sigma': optimal_sigma,
            'max_auc_by_sigma': max_auc_by_sigma,
            'mean_auc_by_sigma': mean_auc_by_sigma,
            'auc_grid': auc_grid,
            'center_positions': center_positions,
            'sigma_values': sigma_values,
            'best_center_by_sigma': best_center_by_sigma
        }
    
    def scan_temporal_profile_with_optimal_sigma(self, dv_mst, dv_vps, time_axis, time_indices, 
                                                choices, sigma_mst, sigma_vps, n_scan_centers=20):
        """
        Scan across time with optimal sigmas to see when each area is most informative.
        """
        # ADD THIS DEBUG BLOCK AT THE START
        print(f"\n[DEBUG] scan_temporal_profile_with_optimal_sigma called")
        print(f"  dv_mst shape: {dv_mst.shape}")
        print(f"  dv_vps shape: {dv_vps.shape}")
        print(f"  time_axis shape: {time_axis.shape}")
        print(f"  time_indices type: {type(time_indices)}")
        print(f"  choices shape: {choices.shape}")
        print(f"  sigma_mst: {sigma_mst}")
        print(f"  sigma_vps: {sigma_vps}")
        print(f"  n_scan_centers: {n_scan_centers}")
        
        # Extract time window
        if isinstance(time_indices, np.ndarray) and time_indices.dtype == bool:
            time_windowed = time_axis[time_indices]
        else:
            time_windowed = time_axis[time_indices]
        
        window_start = time_windowed.min()
        window_end = time_windowed.max()
        
        print(f"  window_start: {window_start:.3f}s")
        print(f"  window_end: {window_end:.3f}s")
        print(f"  time_windowed length: {len(time_windowed)}")
        
        # Define scan positions
        center_positions = np.linspace(window_start, window_end, n_scan_centers)
        print(f"  center_positions: {center_positions[:3]}...{center_positions[-3:]}")
        
        # Initialize result arrays
        auc_mst = np.zeros(n_scan_centers)
        auc_vps = np.zeros(n_scan_centers)
        auc_combined = np.zeros(n_scan_centers)
        accuracy_combined = np.zeros(n_scan_centers)
        coef_mst = np.zeros(n_scan_centers)
        coef_vps = np.zeros(n_scan_centers)
        intercept = np.zeros(n_scan_centers)
        
        # Scan across centers
        print(f"\n[DEBUG] Starting scan loop...")
        for i, center in enumerate(center_positions):
            if i % 5 == 0:  # Print every 5th iteration
                print(f"  Processing center {i+1}/{n_scan_centers}: {center:.3f}s")
            
            # Integrate both areas with their optimal sigmas
            int_mst = self.integrate_dv_gaussian(dv_mst, time_axis, time_indices, center, sigma_mst)
            int_vps = self.integrate_dv_gaussian(dv_vps, time_axis, time_indices, center, sigma_vps)
            
            if i == 0:  # Debug first integration
                print(f"    int_mst: mean={np.mean(int_mst):.3f}, std={np.std(int_mst):.3f}, shape={int_mst.shape}")
                print(f"    int_vps: mean={np.mean(int_vps):.3f}, std={np.std(int_vps):.3f}, shape={int_vps.shape}")
            
            # Individual AUCs
            try:
                auc_m = roc_auc_score(choices, int_mst)
                if auc_m < 0.5:
                    auc_m = 1 - auc_m
                auc_mst[i] = auc_m
            except Exception as e:
                if i == 0:
                    print(f"    ERROR calculating AUC MST: {e}")
                auc_mst[i] = 0.5
            
            try:
                auc_v = roc_auc_score(choices, int_vps)
                if auc_v < 0.5:
                    auc_v = 1 - auc_v
                auc_vps[i] = auc_v
            except Exception as e:
                if i == 0:
                    print(f"    ERROR calculating AUC VPS: {e}")
                auc_vps[i] = 0.5
            
            # Combined logistic model
            X = np.column_stack([int_mst, int_vps])
            try:
                model = LogisticRegression(penalty=None, max_iter=1000)
                model.fit(X, choices)
                y_proba = model.predict_proba(X)[:, 1]
                y_pred = model.predict(X)
                
                auc_combined[i] = roc_auc_score(choices, y_proba)
                accuracy_combined[i] = accuracy_score(choices, y_pred)
                coef_mst[i] = model.coef_[0][0]
                coef_vps[i] = model.coef_[0][1]
                intercept[i] = model.intercept_[0]
                
                if i == 0:  # Debug first model fit
                    print(f"    Combined AUC: {auc_combined[i]:.3f}")
                    print(f"    Accuracy: {accuracy_combined[i]:.3f}")
                    print(f"    Coef MST: {coef_mst[i]:.3f}, Coef VPS: {coef_vps[i]:.3f}")
            except Exception as e:
                if i == 0:
                    print(f"    ERROR fitting model: {e}")
                auc_combined[i] = 0.5
                accuracy_combined[i] = 0.5
                coef_mst[i] = 0.0
                coef_vps[i] = 0.0
                intercept[i] = 0.0
        
        print(f"\n[DEBUG] Scan complete!")
        print(f"  auc_mst range: [{np.min(auc_mst):.3f}, {np.max(auc_mst):.3f}]")
        print(f"  auc_vps range: [{np.min(auc_vps):.3f}, {np.max(auc_vps):.3f}]")
        print(f"  auc_combined range: [{np.min(auc_combined):.3f}, {np.max(auc_combined):.3f}]")
        print(f"  accuracy range: [{np.min(accuracy_combined):.3f}, {np.max(accuracy_combined):.3f}]")
        
        return {
            'center_positions': center_positions,
            'auc_mst': auc_mst,
            'auc_vps': auc_vps,
            'auc_combined': auc_combined,
            'accuracy_combined': accuracy_combined,
            'coef_mst': coef_mst,
            'coef_vps': coef_vps,
            'intercept': intercept,
            'sigma_mst': sigma_mst,
            'sigma_vps': sigma_vps
        }
    
    def fit_logistic_model(self, X, y):
        """
        Fit logistic regression: P(choice) ~ accumulated DVs.
        Returns dict with coefficients, AUC, accuracy, fitted model.
        No regularization (penalty=None) to get interpretable weights.
        """
        model = LogisticRegression(penalty=None, max_iter=1000)
        model.fit(X, y)
        
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        
        auc = roc_auc_score(y, y_proba)
        accuracy = accuracy_score(y, y_pred)
        
        return {
            'coefficients': model.coef_[0],
            'intercept': model.intercept_[0],
            'auc': auc,
            'accuracy': accuracy,
            'model': model
        }
    
    def bootstrap_coefficients(self, X, y, n_iterations=1000, alpha=0.05, seed=42):
        """
        Bootstrap confidence intervals for regression coefficients.
        Resamples trials with replacement n_iterations times.
        Returns: mean, std, CI bounds, all bootstrap samples.
        """
        rng = np.random.RandomState(seed)
        n_trials = X.shape[0]
        n_features = X.shape[1]
        
        bootstrap_coefs = np.zeros((n_iterations, n_features))
        
        for i in range(n_iterations):
            indices = rng.choice(n_trials, size=n_trials, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            try:
                model = LogisticRegression(penalty=None, max_iter=1000)
                model.fit(X_boot, y_boot)
                bootstrap_coefs[i] = model.coef_[0]
            except:
                bootstrap_coefs[i] = np.nan
        
        ci_lower = np.nanpercentile(bootstrap_coefs, 100 * alpha / 2, axis=0)
        ci_upper = np.nanpercentile(bootstrap_coefs, 100 * (1 - alpha / 2), axis=0)
        
        return {
            'coefficients_mean': np.nanmean(bootstrap_coefs, axis=0),
            'coefficients_std': np.nanstd(bootstrap_coefs, axis=0),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'all_coefficients': bootstrap_coefs
        }
    
    def compare_coefficients(self, beta1, beta2, ci1, ci2):
        """
        Test if two coefficients significantly differ.
        Checks if confidence intervals overlap (simple test).
        Returns: difference, whether CIs overlap, significance.
        """
        diff = beta1 - beta2
        ci_overlap = not (ci1[1] < ci2[0] or ci2[1] < ci1[0])
        significant = not ci_overlap
        
        return {
            'difference': diff,
            'ci_overlap': ci_overlap,
            'significant': significant
        }
    
    def grid_search_lambda(self, dv_matrix, time_axis, time_indices, choices, lambda_range=None):
        """
        Grid search for optimal exponential kernel parameter λ.
        Tests λ values to maximize choice prediction AUC.
        Returns: optimal λ, AUC, all AUC scores, tested λ range.
        """
        if lambda_range is None:
            lambda_range = np.linspace(-5, 5, 50)
        
        auc_scores = []
        
        for lam in lambda_range:
            integrated_dv = self.integrate_dv_exponential(dv_matrix, time_axis, time_indices, lam)
            
            X = integrated_dv.reshape(-1, 1)
            try:
                model = LogisticRegression(penalty=None, max_iter=1000)
                model.fit(X, choices)
                y_proba = model.predict_proba(X)[:, 1]
                auc = roc_auc_score(choices, y_proba)
            except:
                auc = 0.5
            
            auc_scores.append(auc)
        
        auc_scores = np.array(auc_scores)
        optimal_idx = np.argmax(auc_scores)
        optimal_lambda = lambda_range[optimal_idx]
        optimal_auc = auc_scores[optimal_idx]
        
        return optimal_lambda, optimal_auc, auc_scores, lambda_range
    
    def grid_search_gaussian(self, dv_matrix, time_axis, time_indices, choices, 
                            center_range=None, sigma_range=None):
        """
        Grid search for optimal Gaussian kernel parameters (center and sigma).
        Tests combinations to maximize choice prediction AUC.
        Returns: optimal center, optimal sigma, AUC, 2D AUC grid, ranges.
        """
        # Get time points in window
        if isinstance(time_indices, np.ndarray) and time_indices.dtype == bool:
            time_windowed = time_axis[time_indices]
        else:
            time_windowed = time_axis[time_indices]
        
        # Default ranges
        if center_range is None:
            center_range = np.linspace(time_windowed.min(), time_windowed.max(), 20)
        if sigma_range is None:
            window_duration = time_windowed.max() - time_windowed.min()
            sigma_range = np.linspace(0.05, window_duration / 2, 15)
        
        auc_grid = np.zeros((len(center_range), len(sigma_range)))
        
        for i, center in enumerate(center_range):
            for j, sigma in enumerate(sigma_range):
                integrated_dv = self.integrate_dv_gaussian(
                    dv_matrix, time_axis, time_indices, center, sigma
                )
                
                X = integrated_dv.reshape(-1, 1)
                try:
                    model = LogisticRegression(penalty=None, max_iter=1000)
                    model.fit(X, choices)
                    y_proba = model.predict_proba(X)[:, 1]
                    auc = roc_auc_score(choices, y_proba)
                    # Handle flipped decoders
                    if auc < 0.5:
                        auc = 1 - auc
                    auc_grid[i, j] = auc
                except:
                    auc_grid[i, j] = 0.5
        
        # Find best parameters
        best_idx = np.unravel_index(np.argmax(auc_grid), auc_grid.shape)
        optimal_center = center_range[best_idx[0]]
        optimal_sigma = sigma_range[best_idx[1]]
        optimal_auc = auc_grid[best_idx]
        
        return optimal_center, optimal_sigma, optimal_auc, auc_grid, center_range, sigma_range
    
    def fit_dual_lambda_model(self, dv_mst, dv_vps, time_axis, time_indices, choices):
        """
        Jointly optimize λ_MST and λ_VPS to maximize choice prediction.
        Fits: Choice ~ β_MST × ∫exp(λ_MST×t)×DV_MST + β_VPS × ∫exp(λ_VPS×t)×DV_VPS.
        Returns: optimal λs, βs, AUC, integrated DVs.
        """
        def objective(params):
            lambda_mst, lambda_vps = params
            
            int_mst = self.integrate_dv_exponential(dv_mst, time_axis, time_indices, lambda_mst)
            int_vps = self.integrate_dv_exponential(dv_vps, time_axis, time_indices, lambda_vps)
            
            X = np.column_stack([int_mst, int_vps])
            try:
                model = LogisticRegression(penalty=None, max_iter=1000)
                model.fit(X, choices)
                y_proba = model.predict_proba(X)[:, 1]
                auc = roc_auc_score(choices, y_proba)
                return -auc
            except:
                return 0.0
        
        result = minimize(objective, x0=[0.0, 0.0], method='Nelder-Mead', options={'maxiter': 1000})
        
        lambda_mst_opt, lambda_vps_opt = result.x
        
        int_mst = self.integrate_dv_exponential(dv_mst, time_axis, time_indices, lambda_mst_opt)
        int_vps = self.integrate_dv_exponential(dv_vps, time_axis, time_indices, lambda_vps_opt)
        X = np.column_stack([int_mst, int_vps])
        
        model = LogisticRegression(penalty=None, max_iter=1000)
        model.fit(X, choices)
        y_proba = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(choices, y_proba)
        
        return {
            'lambda_mst': lambda_mst_opt,
            'lambda_vps': lambda_vps_opt,
            'beta_mst': model.coef_[0][0],
            'beta_vps': model.coef_[0][1],
            'auc': auc,
            'integrated_mst': int_mst,
            'integrated_vps': int_vps
        }
    
    def fit_dual_gaussian_model(self, dv_mst, dv_vps, time_axis, time_indices, choices,
                               center_range=None, sigma_range=None):
        """
        Jointly optimize Gaussian parameters for MST and VPS to maximize choice prediction.
        Fits: Choice ~ β_MST × ∫Gauss(center_MST, σ_MST)×DV_MST + β_VPS × ∫Gauss(center_VPS, σ_VPS)×DV_VPS.
        Returns: optimal parameters, βs, AUC, integrated DVs.
        """
        def objective(params):
            center_mst, sigma_mst, center_vps, sigma_vps = params
            
            # Constrain sigma to be positive
            if sigma_mst <= 0 or sigma_vps <= 0:
                return 0.0
            
            int_mst = self.integrate_dv_gaussian(dv_mst, time_axis, time_indices, center_mst, sigma_mst)
            int_vps = self.integrate_dv_gaussian(dv_vps, time_axis, time_indices, center_vps, sigma_vps)
            
            X = np.column_stack([int_mst, int_vps])
            try:
                model = LogisticRegression(penalty=None, max_iter=1000)
                model.fit(X, choices)
                y_proba = model.predict_proba(X)[:, 1]
                auc = roc_auc_score(choices, y_proba)
                return -auc
            except:
                return 0.0
        
        # Get time range for initial guess
        if isinstance(time_indices, np.ndarray) and time_indices.dtype == bool:
            time_windowed = time_axis[time_indices]
        else:
            time_windowed = time_axis[time_indices]
        
        time_mid = (time_windowed.min() + time_windowed.max()) / 2
        time_span = time_windowed.max() - time_windowed.min()
        
        # Initial guess: center at middle, sigma = 1/4 of window
        x0 = [time_mid, time_span / 4, time_mid, time_span / 4]
        
        # Bounds: centers within window, sigmas positive
        bounds = [
            (time_windowed.min(), time_windowed.max()),  # center_mst
            (0.01, time_span),  # sigma_mst
            (time_windowed.min(), time_windowed.max()),  # center_vps
            (0.01, time_span)   # sigma_vps
        ]
        
        result = minimize(objective, x0=x0, method='L-BFGS-B', bounds=bounds, 
                         options={'maxiter': 1000})
        
        center_mst_opt, sigma_mst_opt, center_vps_opt, sigma_vps_opt = result.x
        
        int_mst = self.integrate_dv_gaussian(dv_mst, time_axis, time_indices, center_mst_opt, sigma_mst_opt)
        int_vps = self.integrate_dv_gaussian(dv_vps, time_axis, time_indices, center_vps_opt, sigma_vps_opt)
        X = np.column_stack([int_mst, int_vps])
        
        model = LogisticRegression(penalty=None, max_iter=1000)
        model.fit(X, choices)
        y_proba = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(choices, y_proba)
        
        return {
            'center_mst': center_mst_opt,
            'sigma_mst': sigma_mst_opt,
            'center_vps': center_vps_opt,
            'sigma_vps': sigma_vps_opt,
            'beta_mst': model.coef_[0][0],
            'beta_vps': model.coef_[0][1],
            'auc': auc,
            'integrated_mst': int_mst,
            'integrated_vps': int_vps
        }