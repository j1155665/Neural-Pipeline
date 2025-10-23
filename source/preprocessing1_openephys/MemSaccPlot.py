import os
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, ttest_ind
import glob
from tkinter import Tk, filedialog
import pandas as pd

class NeuralDataLoader:
    def __init__(self, base_dir=None, electrode_id=None, kilo_dir=None, sacc_file=None, verbose=True, tdi_threshold = 0.45):
        self.verbose = verbose
        if base_dir is None:
            try:
                base_dir = self.select_folder()
            except:
                raise RuntimeError("No base_dir provided and GUI folder picker failed in Jupyter. Please pass a folder path.")
        self.base_dir = base_dir
        if self.verbose:
            print(f"Selected base directory: {self.base_dir}")

        self.kilosort_dir = kilo_dir
        self.sacc_file = sacc_file
        self.tdi_threshold = tdi_threshold
        self._initialize_attributes()
        self.load_all(electrode_id)

    def _initialize_attributes(self):
        """Initialize all data attributes to None"""
        attrs = ['continuous_data', 'continuous_timestamps', 'ttl_data', 'ttl_timestamps', 
                'channel_positions', 'spike_times', 'cluster_id', 'current_cluster_spikes']
        for attr in attrs:
            setattr(self, attr, None)

    def _print(self, message):
        """Conditional printing based on verbose flag"""
        if self.verbose:
            print(message)

    def _get_spike_times(self):
        """Helper method to get appropriate spike times"""
        if hasattr(self, 'current_cluster_spikes') and self.current_cluster_spikes is not None:
            return self.continuous_timestamps[self.current_cluster_spikes]
        return self.spike_times

    def _find_top_bottom_conditions(self, avg_firing_rates, n=3):
        """Helper method to find top and bottom N conditions"""
        if not avg_firing_rates:
            return [], []
        sorted_rates = sorted(avg_firing_rates.items(), key=lambda x: x[1], reverse=True)
        top_conditions = [cond for cond, rate in sorted_rates[:n] if not np.isnan(rate)]
        bottom_conditions = [cond for cond, rate in sorted_rates[-n:] if not np.isnan(rate)]
        return top_conditions, bottom_conditions

    def _calculate_psth_sliding_window(self, spike_times, trial_mask, stim_on, 
                                    time_bins, window_size=0.1, smooth_sigma_ms=0.5):
        """Helper method for PSTH calculation with sliding window and Gaussian smoothing"""
        from scipy import ndimage
        
        aligned_counts = []
        for trial_idx in np.where(trial_mask)[0]:
            t0 = stim_on[trial_idx]
            spikes = spike_times - t0
            trial_counts = []
            for t in time_bins:
                in_window = (spikes >= t - window_size) & (spikes < t)
                rate = np.sum(in_window) / window_size
                trial_counts.append(rate)
            aligned_counts.append(trial_counts)
        
        if not aligned_counts:
            return np.array([]).reshape(0, len(time_bins))
        
        aligned_counts = np.array(aligned_counts)
        
        # Apply Gaussian smoothing if requested
        if smooth_sigma_ms > 0:
            # Convert sigma from ms to time bins
            step_size_ms = np.mean(np.diff(time_bins)) * 1000  # Convert to ms
            sigma_bins = smooth_sigma_ms / step_size_ms
            
            # Apply Gaussian filter along time axis (axis=1) for each trial
            smoothed_counts = np.zeros_like(aligned_counts)
            for trial_idx in range(aligned_counts.shape[0]):
                smoothed_counts[trial_idx, :] = ndimage.gaussian_filter1d(
                    aligned_counts[trial_idx, :], sigma=sigma_bins, mode='nearest'
                )
            
            return smoothed_counts
        
        return aligned_counts

    def select_folder(self):
        """Open a dialog to select a folder"""
        root = Tk()
        root.withdraw()
        folder = filedialog.askdirectory(title="Select experiment folder")
        if not folder:
            raise ValueError("No folder selected")
        return folder

    def load_continuous_data(self, num_channels=384): 
        """Load continuous.dat and timestamps.npy"""
        cont_dir = os.path.join(self.base_dir, "continuous", "Neuropix-PXI-100.ProbeA-AP")
        cont_file = os.path.join(cont_dir, "continuous.dat")
        ts_file = os.path.join(cont_dir, "timestamps.npy")

        if not (os.path.exists(cont_file) and os.path.exists(ts_file)):
            raise FileNotFoundError("Continuous data or timestamps not found")

        self._print("Loading continuous data...")
        self.continuous_data = np.fromfile(cont_file, dtype=np.int16).reshape((-1, num_channels))
        self.continuous_timestamps = np.load(ts_file)

    def load_spike_data(self, electrode_id):
        """Load spike timestamps for a specific electrode."""
        spike_base = os.path.join(self.base_dir, "spikes")
        probe_dirs = glob.glob(os.path.join(spike_base, "Spike_Detector-*.ProbeA-AP"))
        if len(probe_dirs) != 1:
            raise RuntimeError(f"Expected one ProbeA-AP directory, found {len(probe_dirs)}")

        probe_dir = probe_dirs[0]
        pattern = os.path.join(probe_dir, f"* {electrode_id}")
        candidates = glob.glob(pattern)

        valid_prefixes = ["Tetrode", "Stereotrode", "Electrode"]
        filtered = [d for d in candidates if any(os.path.basename(d).startswith(prefix) for prefix in valid_prefixes)]

        if len(filtered) != 1:
            raise RuntimeError(f"Expected one valid electrode folder, found {len(filtered)}")

        spike_dir = filtered[0]
        timestamps_file = os.path.join(spike_dir, "timestamps.npy")
        cluster_file = os.path.join(spike_dir, "clusters.npy")

        if not os.path.exists(timestamps_file):
            raise FileNotFoundError(f"No spike timestamps found for Electrode {electrode_id}")

        self._print(f"Loading spikes from Electrode {electrode_id}...")
        self.spike_times = np.load(timestamps_file)
        self.cluster_id = np.load(cluster_file)

    def load_kilosort_data(self):
        """Load Kilosort output files"""
        self._print("Loading kilosort data...")
        kilosort_dir = self.kilosort_dir
        
        # Load basic spike data
        self.spike_times = np.load(os.path.join(kilosort_dir, "spike_times.npy"))      
        self.cluster_id = np.load(os.path.join(kilosort_dir, "spike_clusters.npy"))  
        self.cluster_info_file = os.path.join(kilosort_dir, "cluster_KSLabel.tsv")
        
        # Load position files if available
        position_files = {
            'templates': os.path.join(kilosort_dir, "templates.npy"),
            'channel_map': os.path.join(kilosort_dir, "channel_map.npy"),
            'channel_positions': os.path.join(kilosort_dir, "channel_positions.npy")
        }
        
        if all(os.path.exists(f) for f in position_files.values()):
            self.templates = np.load(position_files['templates'])
            self.channel_map = np.load(position_files['channel_map'])
            self.channel_positions = np.load(position_files['channel_positions'])
            self.cluster_depth = self._get_cluster_depths()
        else:
            if self.verbose:
                print("Warning: Channel position files not found. Cluster depths not available.")
            self.cluster_depth = None

    def _get_cluster_depths(self, threshold=0.1):
        """Calculate cluster depths based on channel positions"""
        unique_clusters = np.unique(self.cluster_id)
        cluster_depths = {}
        
        for cluster_id in unique_clusters:
            if cluster_id < len(self.templates):
                template = self.templates[cluster_id]
                peak_amplitudes = np.max(np.abs(template), axis=0)
                max_amp = np.max(peak_amplitudes)
                
                if max_amp > 0:
                    significant_channels = np.where(peak_amplitudes > threshold * max_amp)[0]
                    positions = self.channel_positions[significant_channels]
                    weights = peak_amplitudes[significant_channels]
                    weighted_depth = np.average(positions[:, 1], weights=weights)
                    cluster_depths[cluster_id] = weighted_depth
                else:
                    cluster_depths[cluster_id] = np.nan
        
        return cluster_depths

    def load_ttl_data(self):
        """Load TTL data from appropriate source"""
        if self.kilosort_dir is None:
            # Load from OpenEphys structure
            ttl_dir = os.path.join(self.base_dir, "events", "NI-DAQmx-107.PXIe-6341", "TTL")
            ttl_file = os.path.join(ttl_dir, "full_words.npy")
            ts_file = os.path.join(ttl_dir, "timestamps.npy")
            
            if not (os.path.exists(ttl_file) and os.path.exists(ts_file)):
                raise FileNotFoundError("TTL data or timestamps not found")
        else:
            # Load from kilosort directory structure
            file_patterns = {
                'ttl': "*ttl.npy",
                'timestamps': "*ttltimestamps.npy", 
                'blocks': "*ttlblocks.npy",
                'ap_timestamps': "*APtimestamps.npy"
            }
            
            files = {}
            for key, pattern in file_patterns.items():
                matches = glob.glob(os.path.join(self.base_dir, pattern))
                if len(matches) != 1:
                    raise FileNotFoundError(f"Expected exactly 1 {key} file, found {len(matches)}")
                files[key] = matches[0]
            
            ttl_file = files['ttl']
            ts_file = files['timestamps']

        self.ttl_data = np.load(ttl_file)
        self.ttl_timestamps = np.load(ts_file)
        
        if self.kilosort_dir is not None:
            self.ttl_blocks = np.load(files['blocks'])
            self.continuous_timestamps = np.load(files['ap_timestamps'])

    def load_probe_config(self):
        """Parse settings.xml to get channel positions"""
        parent_dir = os.path.dirname(os.path.dirname(self.base_dir))
        settings_file = os.path.join(parent_dir, "settings.xml")
        if not os.path.exists(settings_file):
            raise FileNotFoundError(f"settings.xml not found in {parent_dir}")
        
        tree = ET.parse(settings_file)
        root = tree.getroot()
        probe_node = root.find(".//NP_PROBE")

        xpos_node = probe_node.find("ELECTRODE_XPOS")
        ypos_node = probe_node.find("ELECTRODE_YPOS")

        xpos = {int(k[2:]): int(v) for k, v in xpos_node.attrib.items() if k.startswith("CH")}
        ypos = {int(k[2:]): int(v) for k, v in ypos_node.attrib.items() if k.startswith("CH")}

        ch_list = sorted(set(xpos) | set(ypos))
        self.channel_positions = np.array([[ch, xpos.get(ch, np.nan), ypos.get(ch, np.nan)] for ch in ch_list])

    def load_all(self, electrode_id):
        """Load all data components"""
        if electrode_id is None:
            if self.kilosort_dir is None:
                self.load_continuous_data()
                self.load_probe_config()
            else:
                self.load_kilosort_data()
        else:
            self.load_spike_data(electrode_id)
            self.load_probe_config()

        self.load_ttl_data()

    def get_spike_times(self, channel_idx, threshold=-50, refractory_ms=1):
        """Quick spike detection on one channel"""
        fs = 1 / np.mean(np.diff(self.continuous_timestamps))
        signal = self.continuous_data[:, channel_idx]

        crossings = np.where((signal[:-1] > threshold) & (signal[1:] <= threshold))[0] + 1
        refractory_samples = int(refractory_ms * fs / 1000)
        
        spike_times = []
        last_spike = -np.inf
        for t in crossings:
            if t - last_spike > refractory_samples:
                spike_times.append(t)
                last_spike = t

        spike_times = np.array(spike_times)
        spike_times_sec = self.continuous_timestamps[spike_times]
        self.spike_times = spike_times_sec
        self._print(f"Detected {len(spike_times_sec)} spikes on channel {channel_idx}")
        return self.spike_times

    def parse_ttl_events(self):
        """Parse TTL signals to extract trial events"""
        # Event codes and names
        event_codes = [1, 2, 3, 4, 4, 6, 9, 10]  # TRIAL, FIX, FIXATION, STIMON, STIMOFF, SACC, REWARD, BREAKFIX
        event_names = ['TRIAL', 'FIX', 'FIXATION', 'STIMON', 'STIMOFF', 'SACC', 'REWARD', 'BREAKFIX']

        # Clean up TTL values and filter by block if needed
        ttl_values = np.where(self.ttl_data >= 256, self.ttl_data - 256, self.ttl_data)
        ttl_times = self.ttl_timestamps

        if hasattr(self, 'sacc_file') and self.sacc_file is not None:
            valid_indices = np.where(self.ttl_blocks == self.sacc_file)[0]
            ttl_values = ttl_values[valid_indices]
            ttl_times = ttl_times[valid_indices]
            self._print(f"Filtering TTL data for block {self.sacc_file}")

        # Find trial indices
        trial_indices = np.where(ttl_values == 1)[0]  # TRIAL events
        fix_indices = np.where(ttl_values == 2)[0]    # FIX events
        idx_diffs = np.diff(trial_indices) > 10
        self.trial_indices = np.concatenate(([trial_indices[0]], trial_indices[1:][idx_diffs]))

        # Initialize event data
        event_data = {name: [] for name in event_names}
        event_data.update({'condition': [], 'goodtrial': []})

        # Process each trial
        for i, trial_idx in enumerate(trial_indices):
            trial_start = trial_indices[i-1] if i != 0 else (fix_indices[0] if fix_indices.size > 0 else 0)
            
            trial_ttls = ttl_values[trial_start:trial_idx]
            trial_times = ttl_times[trial_start:trial_idx]

            # Extract event timestamps
            trial_events = {}
            for name, code in zip(event_names, event_codes):
                matching = np.where(trial_ttls == code)[0]
                if matching.size > 0:
                    trial_events[name] = trial_times[matching[-1] if name == 'STIMOFF' else matching[0]]
                else:
                    trial_events[name] = np.nan
                event_data[name].append(trial_events[name])

            # Extract condition code and mark good trials
            cond_idx = np.where(trial_ttls > 100)[0]
            cond_code = trial_ttls[cond_idx[0]] - 100 if cond_idx.size > 0 else np.nan
            event_data['condition'].append(cond_code)
            event_data['goodtrial'].append(0 if not np.isnan(trial_events.get('BREAKFIX', np.nan)) else 1)

        # Convert to numpy arrays
        for key in event_data:
            event_data[key] = np.array(event_data[key])

        self.event_data = event_data
        self._print(f"Parsed {len(trial_indices)} trials, {np.sum(event_data['goodtrial'])} good trials.")
        return self.event_data

    def avg_firing_rate(self, align_info):
        """Calculate average firing rates per condition"""
        event_name, pre, post = align_info['event'], align_info['pre'], align_info['post']
        
        conditions = self.event_data['condition']
        align_event = self.event_data[event_name]
        good_trials = self.event_data['goodtrial']
        spike_times = self._get_spike_times()

        unique_conditions = np.unique(conditions[~np.isnan(conditions)]).astype(int)
        avg_firing_rates = {}
        var_firing_rates = {}

        for cond in unique_conditions:
            trial_inds = np.where((conditions == cond) & (good_trials == 1))[0]
            rates = []
            
            for idx in trial_inds:
                start, end = align_event[idx] + pre, align_event[idx] + post
                if np.isnan(start) or np.isnan(end) or end <= start:
                    continue
                
                spikes_in_interval = np.sum((spike_times >= start) & (spike_times <= end))
                rates.append(spikes_in_interval / (end - start))

            if rates:
                avg_firing_rates[cond] = np.mean(rates)
                var_firing_rates[cond] = np.var(rates)
            else:
                avg_firing_rates[cond] = np.nan
                var_firing_rates[cond] = np.nan

        # Compute TDI
        rates_list = list(avg_firing_rates.values())
        r_max, r_min = np.nanmax(rates_list), np.nanmin(rates_list)
        sse = np.nansum([var_firing_rates[c] * len(np.where((conditions == c) & (good_trials == 1))[0]) 
                        for c in unique_conditions])
        N, M = int(np.sum(good_trials)), len(unique_conditions)

        if np.isnan(r_max) or np.isnan(r_min) or (r_max - r_min) == 0 or N <= M or sse < 0:
            tdi = np.nan
        else:
            tdi = (r_max - r_min) / (r_max - r_min + 2 * np.sqrt(sse / (N - M)))

        return avg_firing_rates, var_firing_rates, tdi

    def plot_avg_firing_rate_heatmap(self, align_info, avg_firing_rates, eccs, angles, ax=None):
        """Plot spatial firing rate heatmap"""
        if ax is None:
            ax = plt.gca()

        rates = np.array(list(avg_firing_rates.values()))
        if len(rates) == 0 or np.all(np.isnan(rates)):
            self._print("No valid firing rates to plot.")
            return

        x = eccs * np.cos(np.deg2rad(angles))
        y = eccs * np.sin(np.deg2rad(angles))

        ax.set_xlim(-12, 12)
        ax.set_ylim(-12, 12)
        ax.set_facecolor(plt.cm.viridis(0.0))

        scatter = ax.scatter(x, y, c=rates, cmap='viridis', s=100, edgecolor=None, marker='s')
        plt.colorbar(scatter, ax=ax, label='Firing Rate')
        ax.set_xlabel('X position (deg)')
        ax.set_ylabel('Y position (deg)')
        ax.set_title("Memory Saccade Heatmap")
        ax.grid(False)
        ax.axis('equal')

    def plot_psth(self, avg_firing_rates, compare_position, ax=None):
        """Plot PSTH comparing top vs bottom conditions"""
        if ax is None:
            ax = plt.gca()

        # Time parameters
        window_size, step_size = 0.1, 0.02
        time_start, time_end = -0.1, 1.2
        time_bins = np.arange(time_start, time_end - window_size + step_size, step_size)

        # Get data
        conditions = np.array(self.event_data['condition'])
        stim_on = np.array(self.event_data['STIMON'])
        good_trials = np.array(self.event_data['goodtrial']) == 1
        spike_times = self._get_spike_times()

        # Find top and bottom conditions
        top_conditions, bottom_conditions = self._find_top_bottom_conditions(avg_firing_rates, compare_position)
        
        max_trials = np.isin(conditions, top_conditions) & good_trials
        min_trials = np.isin(conditions, bottom_conditions) & good_trials

        # Calculate PSTHs
        max_spike_mat = self._calculate_psth_sliding_window(spike_times, max_trials, stim_on, time_bins, window_size)
        min_spike_mat = self._calculate_psth_sliding_window(spike_times, min_trials, stim_on, time_bins, window_size)

        if max_spike_mat.size == 0 or min_spike_mat.size == 0:
            self._print("No valid trials for PSTH")
            return

        # Calculate means and SEMs
        max_mean = np.mean(max_spike_mat, axis=0)
        max_sem = np.std(max_spike_mat, axis=0) / np.sqrt(max_spike_mat.shape[0])
        min_mean = np.mean(min_spike_mat, axis=0)
        min_sem = np.std(min_spike_mat, axis=0) / np.sqrt(min_spike_mat.shape[0])

        # Statistical test
        if compare_position == 1 and max_spike_mat.shape[0] == min_spike_mat.shape[0]:
            _, p_vals = ttest_rel(max_spike_mat, min_spike_mat, axis=0)
        else:
            _, p_vals = ttest_ind(max_spike_mat, min_spike_mat, axis=0, equal_var=False)

        # Plot
        ax.plot(time_bins, max_mean, color='blue', alpha=0.5, label=f'Max cond ({top_conditions})')
        ax.fill_between(time_bins, max_mean - max_sem, max_mean + max_sem, color='blue', alpha=0.3)
        ax.plot(time_bins, min_mean, color='blue', linestyle='--', alpha=0.5, label=f'Min cond ({bottom_conditions})')
        ax.fill_between(time_bins, min_mean - min_sem, min_mean + min_sem, color='blue', alpha=0.05)

        # Add significance markers
        sig_mask = p_vals < 0.05
        if np.any(sig_mask):
            sig_y = max(np.max(max_mean + max_sem), np.max(min_mean + min_sem)) + 1
            for t in time_bins[sig_mask]:
                ax.hlines(sig_y, t, t + step_size, color='black', linewidth=2)

        # Add reference lines
        ax.axvspan(0, 0.1, color='gray', alpha=0.2)
        ax.axvline(0.9, color='purple', linestyle=':')
        ax.set_xlabel('Time from STIMON (s)')
        ax.set_ylabel('Firing rate (Hz)')
        ax.set_title(f"Top conditions: {top_conditions}")
        ax.set_xlim([-0.1, 1])

    def process_all_units_comprehensive(self, align_info):
        """
        Process all units and create comprehensive PSTH heatmap
        This method handles everything: data collection + heatmap creation
        """
        # Parse TTL events if not already done
        if not hasattr(self, 'event_data'):
            self.parse_ttl_events()
        
        # Load cluster info
        cluster_info = pd.read_csv(self.cluster_info_file, sep='\t')
        all_clusters = cluster_info['cluster_id'].values
        cluster_labels = dict(zip(cluster_info['cluster_id'], cluster_info['KSLabel']))
        
        # Initialize arrays to store all data
        tdi_all = np.full(len(all_clusters), np.nan)
        fr_all = np.full(len(all_clusters), np.nan)
        depth_all = np.full(len(all_clusters), np.nan)
        cluster_data_all = []
        
        self._print(f"Processing {len(all_clusters)} clusters...")
        
        for idx, cluster_id in enumerate(all_clusters):
            cluster_label = cluster_labels[cluster_id]
            
            try:
                cluster_spikes = self.spike_times[self.cluster_id == cluster_id]
                self.current_cluster_spikes = cluster_spikes
                
                avg_firing_rates, _, tdi = self.avg_firing_rate(align_info)
                mean_fr = np.nanmean(list(avg_firing_rates.values()))
                
                # Get depth
                depth = np.nan
                if self.cluster_depth is not None and cluster_id in self.cluster_depth:
                    depth = self.cluster_depth[cluster_id]
                
                # Store in arrays
                tdi_all[idx] = tdi
                fr_all[idx] = mean_fr
                depth_all[idx] = depth
                
                # Store detailed cluster data
                cluster_data = {
                    'cluster_id': cluster_id,
                    'cluster_label': cluster_label,
                    'avg_firing_rates': avg_firing_rates,
                    'tdi': tdi,
                    'mean_fr': mean_fr,
                    'depth': depth,
                    'index': idx
                }
                cluster_data_all.append(cluster_data)
                
            except Exception as e:
                if self.verbose:
                    print(f"Error processing cluster {cluster_id}: {e}")
        
        # Create the comprehensive PSTH heatmap
        top_matrix, bottom_matrix, unit_info, time_bins = self.create_comprehensive_psth_heatmap(
            cluster_data_all, self.base_dir, self.sacc_file
        )
        
        # Print summary statistics
        self._print(f"Total clusters: {len(cluster_data_all)}")
        self._print(f"Valid TDI values: {np.sum(~np.isnan(tdi_all))}")
        self._print(f"Valid FR values: {np.sum(~np.isnan(fr_all))}")
        self._print(f"Valid depth values: {np.sum(~np.isnan(depth_all))}")
        self.tdi_all = tdi_all
        
        if unit_info:
            self._print(f"Units in heatmap: {len(unit_info)}")
            depth_values = [u['depth'] for u in unit_info if not np.isnan(u['depth'])]
            if depth_values:
                self._print(f"Depth range: {np.min(depth_values):.0f} - {np.max(depth_values):.0f}")
            if time_bins is not None:
                self._print(f"Time bins: {len(time_bins)} bins from {time_bins[0]:.2f}s to {time_bins[-1]:.2f}s")
        
        return tdi_all, fr_all, depth_all, cluster_data_all, top_matrix, bottom_matrix, unit_info, time_bins
    
    def create_comprehensive_psth_heatmap(self, cluster_data_all, base_dir, sacc_file):
        """Create comprehensive PSTH heatmap for all units"""
        # Filter valid units and sort by depth
        valid_units = [unit for unit in cluster_data_all 
                    if not np.isnan(unit['depth']) and len(unit['avg_firing_rates']) > 0]
        
        if not valid_units:
            self._print("No valid units found for plotting")
            return None, None, None, None
        
        valid_units.sort(key=lambda x: x['depth'], reverse=False)
        n_units = len(valid_units)

        # Time parameters
        window_size, step_size = 0.1, 0.02
        time_start, time_end = -0.1, 1.1
        time_bins = np.arange(time_start, time_end - window_size + step_size, step_size)
        n_time_bins = len(time_bins)

        # Initialize matrices
        top_3_psth_matrix = np.full((n_units, n_time_bins), np.nan)
        bottom_3_psth_matrix = np.full((n_units, n_time_bins), np.nan)

        # Get trial data
        conditions = np.array(self.event_data['condition'])
        stim_on = np.array(self.event_data['STIMON'])
        good_trials = np.array(self.event_data['goodtrial']) == 1

        unit_info = []
        self._print(f"Processing {n_units} units for PSTH heatmap...")

        for unit_idx, unit_data in enumerate(valid_units):
            try:
                # Get spike times for this unit
                cluster_id = unit_data['cluster_id']
                cluster_spikes = self.spike_times[self.cluster_id == cluster_id]
                spike_times = self.continuous_timestamps[cluster_spikes]

                # Find top/bottom conditions
                top_3_conditions, bottom_3_conditions = self._find_top_bottom_conditions(
                    unit_data['avg_firing_rates'], 3)

                if top_3_conditions and bottom_3_conditions:
                    # Create trial masks
                    top_trials = np.isin(conditions, top_3_conditions) & good_trials
                    bottom_trials = np.isin(conditions, bottom_3_conditions) & good_trials

                    # Calculate PSTHs
                    top_psth_mat = self._calculate_psth_sliding_window(
                        spike_times, top_trials, stim_on, time_bins, window_size)
                    bottom_psth_mat = self._calculate_psth_sliding_window(
                        spike_times, bottom_trials, stim_on, time_bins, window_size)

                    # Store mean PSTHs
                    if top_psth_mat.size > 0:
                        top_3_psth_matrix[unit_idx, :] = np.mean(top_psth_mat, axis=0)
                    if bottom_psth_mat.size > 0:
                        bottom_3_psth_matrix[unit_idx, :] = np.mean(bottom_psth_mat, axis=0)

                # Store unit info
                unit_info.append({
                    'cluster_id': unit_data['cluster_id'],
                    'depth': unit_data['depth'],
                    'tdi': unit_data['tdi'],
                    'mean_fr': unit_data['mean_fr'],
                    'cluster_label': unit_data['cluster_label'],
                    'top_conditions': top_3_conditions,
                    'bottom_conditions': bottom_3_conditions
                })

            except Exception as e:
                if self.verbose:
                    print(f"Error processing unit {unit_data['cluster_id']}: {e}")
                unit_info.append({
                    'cluster_id': unit_data['cluster_id'],
                    'depth': unit_data['depth'],
                    'tdi': np.nan, 'mean_fr': np.nan,
                    'cluster_label': 'error',
                    'top_conditions': [], 'bottom_conditions': []
                })

        # Normalization using bottom 3 conditions as baseline
        self._print("Calculating delta FR / baseline FR normalization...")
        
        top_3_psth_matrix_norm = np.full_like(top_3_psth_matrix, np.nan)
        bottom_3_psth_matrix_norm = np.full_like(bottom_3_psth_matrix, np.nan)
        
        for unit_idx in range(n_units):
            top_row = top_3_psth_matrix[unit_idx, :]
            bottom_row = bottom_3_psth_matrix[unit_idx, :]
            
            # Use bottom 3 conditions as baseline
            bottom_baseline = np.nanmean(bottom_3_psth_matrix)

            epsilon = 1e-6 
            if np.isnan(bottom_baseline) or bottom_baseline <= 0:
                bottom_baseline = epsilon

            top_3_psth_matrix_norm[unit_idx, :] = (top_row - bottom_baseline) / bottom_baseline
            bottom_3_psth_matrix_norm[unit_idx, :] = (bottom_row - bottom_baseline) / bottom_baseline

        # Create unified matrices for each TDI category (same size as original, filled with NaN where not applicable)
        high_tdi_top_matrix_unified = np.full_like(top_3_psth_matrix_norm, np.nan)
        high_tdi_bottom_matrix_unified = np.full_like(bottom_3_psth_matrix_norm, np.nan)
        low_tdi_top_matrix_unified = np.full_like(top_3_psth_matrix_norm, np.nan)
        low_tdi_bottom_matrix_unified = np.full_like(bottom_3_psth_matrix_norm, np.nan)

        # Fill matrices based on TDI categories
        n_high_tdi = 0
        n_low_tdi = 0
        tdi_threshold = self.tdi_threshold
    
        for unit_idx in range(n_units):
            unit_tdi = unit_info[unit_idx]['tdi']
            if not np.isnan(unit_tdi) and unit_tdi > tdi_threshold:
                high_tdi_top_matrix_unified[unit_idx, :] = top_3_psth_matrix_norm[unit_idx, :]
                high_tdi_bottom_matrix_unified[unit_idx, :] = bottom_3_psth_matrix_norm[unit_idx, :]
                n_high_tdi += 1
            else:
                low_tdi_top_matrix_unified[unit_idx, :] = top_3_psth_matrix_norm[unit_idx, :]
                low_tdi_bottom_matrix_unified[unit_idx, :] = bottom_3_psth_matrix_norm[unit_idx, :]
                n_low_tdi += 1

        # Create 2x2 subplot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, max(8, n_units * 0.03)))

        # Set color scale
        all_data = np.concatenate([
            top_3_psth_matrix_norm[~np.isnan(top_3_psth_matrix_norm)],
            bottom_3_psth_matrix_norm[~np.isnan(bottom_3_psth_matrix_norm)]
        ])

        if len(all_data) > 0:
            vmax = np.percentile(all_data, 95)
            vmin = np.percentile(all_data, 5)
            print(f"95th percentile: {np.percentile(all_data, 95):.2f}, 5th percentile: {np.percentile(all_data, 5):.2f}")
        else:
            vmin, vmax = -1, 1

        # Get overall depth range for all units
        all_depths = [unit['depth'] for unit in unit_info]
        depth_min = min(all_depths)
        depth_max = max(all_depths)
        extent_unified = [time_start, time_end, depth_min, depth_max]

        # Plot high TDI units (top row) - using unified matrices
        im1 = ax1.imshow(high_tdi_top_matrix_unified, aspect='auto', cmap='viridis', 
                        interpolation='nearest', extent=extent_unified, vmin=vmin, vmax=vmax, origin='lower')
        im3 = ax3.imshow(high_tdi_bottom_matrix_unified, aspect='auto', cmap='viridis', 
                        interpolation='nearest', extent=extent_unified, vmin=vmin, vmax=vmax, origin='lower')
        
        ax1.set_title(f'High TDI (>{tdi_threshold}) - Top 3 Conditions (n={n_high_tdi})', fontweight='bold')
        ax3.set_title(f'High TDI (>{tdi_threshold}) - Bottom 3 Conditions (n={n_high_tdi})', fontweight='bold')

        # Plot low TDI units (bottom row) - using unified matrices
        im2 = ax2.imshow(low_tdi_top_matrix_unified, aspect='auto', cmap='viridis', 
                        interpolation='nearest', extent=extent_unified, vmin=vmin, vmax=vmax, origin='lower')
        im4 = ax4.imshow(low_tdi_bottom_matrix_unified, aspect='auto', cmap='viridis', 
                        interpolation='nearest', extent=extent_unified, vmin=vmin, vmax=vmax, origin='lower')
        
        ax2.set_title(f'Low TDI (≤{tdi_threshold}) - Top 3 Conditions (n={n_low_tdi})', fontweight='bold')
        ax4.set_title(f'Low TDI (≤{tdi_threshold}) - Bottom 3 Conditions (n={n_low_tdi})', fontweight='bold')

        # Set unified y-ticks for all subplots and add real depth markers every 20 lines
        depth_ticks = np.linspace(depth_min, depth_max, 10)
        
        for ax in [ax1, ax2, ax3, ax4]:
            # Create y-tick positions and labels using real depths every 20 units
            ytick_positions = []
            ytick_labels = []
            
            for i in range(0, n_units, 50):
                if i < len(unit_info):
                    real_depth = unit_info[i]['depth']
                    ytick_positions.append(real_depth)
                    ytick_labels.append(f"{real_depth:.0f}")
            
            # Set the y-ticks to show real depths
            ax.set_yticks(ytick_positions)
            ax.set_yticklabels(ytick_labels)

        # Add colorbars
        plt.colorbar(im1, ax=ax1, shrink=0.6, label='ΔFR / Bottom 3 conditions')
        plt.colorbar(im2, ax=ax2, shrink=0.6, label='ΔFR / Bottom 3 conditions')
        plt.colorbar(im3, ax=ax3, shrink=0.6, label='ΔFR / Bottom 3 conditions')
        plt.colorbar(im4, ax=ax4, shrink=0.6, label='ΔFR / Bottom 3 conditions')

        # Add labels and formatting to all axes
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xlabel('Time from STIMON (s)', fontsize=10)
            ax.set_ylabel('Depth (μm)', fontsize=10)
            
            # Add stimulus period as shaded region
            ax.axvspan(0, 0.1, color='white', alpha=0.3)
            
            stimulus_center_axis = (0.05 - (-0.1)) / (1.1 - (-0.1))  # ≈ 0.125
            go_cue_axis = (0.9 - (-0.1)) / (1.1 - (-0.1))           # ≈ 0.833
            
            ax.text(stimulus_center_axis, 1, 'Stimulus', transform=ax.transAxes, 
                    ha='center', va='bottom', fontsize=10, color='black')
            ax.text(go_cue_axis, 1, 'To go', transform=ax.transAxes, 
                    ha='center', va='bottom', fontsize=10, color='black')
            
            # Keep the saccade line
            ax.axvline(0.9, color='purple', linestyle=':', linewidth=3, alpha=0.8)

        print(f"High TDI units: {n_high_tdi}, Low TDI units: {n_low_tdi}")

        plt.tight_layout()

        # Save figure
        save_path = os.path.join(base_dir, f"psth_heatmap_sacc{sacc_file}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        self._print(f"PSTH heatmap saved to: {save_path}")                              
        
        return top_3_psth_matrix_norm, bottom_3_psth_matrix_norm, unit_info, time_bins
    
    def create_comprehensive_tdi_heatmap(self, event_name='STIMON'):
        """
        Create comprehensive TDI heatmap over time, sorted by depth
        """
        # Parse TTL events if not already done
        if not hasattr(self, 'event_data'):
            self.parse_ttl_events()
        
        # Load cluster info
        cluster_info = pd.read_csv(self.cluster_info_file, sep='\t')
        all_clusters = cluster_info['cluster_id'].values
        cluster_labels = dict(zip(cluster_info['cluster_id'], cluster_info['KSLabel']))
        
        # Get event data
        conditions = np.array(self.event_data['condition'])
        align_event = np.array(self.event_data['STIMON'])
        good_trials = np.array(self.event_data['goodtrial']) == 1
        all_depths = self._get_cluster_depths()

        # Collect valid units with depth info
        cluster_data_all = []
        for cluster_id in all_clusters:
            try:
                # Get depth from channel map
                depth = all_depths[cluster_id]
                if not np.isnan(depth):
                    cluster_data_all.append({
                        'cluster_id': cluster_id,
                        'depth': depth,
                        'cluster_label': cluster_labels.get(cluster_id, 'unknown')
                    })
            except Exception as e:
                if self.verbose:
                    print(f"Error processing cluster {cluster_id}: {e}")
        
        # Filter and sort by depth
        valid_units = [unit for unit in cluster_data_all if not np.isnan(unit['depth'])]
        if not valid_units:
            self._print("No valid units found for plotting")
            return None
        
        valid_units.sort(key=lambda x: x['depth'], reverse=False)
        n_units = len(valid_units)
        
        # Time parameters
        window_size, step_size = 0.1, 0.02
        time_start, time_end = -0.1, 1.1
        time_bins = np.arange(time_start, time_end - window_size + step_size, step_size)
        n_time_bins = len(time_bins)
        
        # Define baseline period (before stimulus onset)
        baseline_start, baseline_end = -0.1, 0.0  # 100ms before stimulus
        baseline_indices = np.where((time_bins >= baseline_start) & (time_bins < baseline_end))[0]
        
        # Initialize matrices
        tdi_matrix = np.full((n_units, n_time_bins), np.nan)
        tdi_baseline_corrected = np.full((n_units, n_time_bins), np.nan)
        
        # Get unique conditions
        unique_conditions = np.unique(conditions[good_trials & ~np.isnan(conditions)]).astype(int)
        
        # Process each unit
        for unit_idx, unit_data in enumerate(valid_units):
            try:
                # Get spike times for this unit
                print(f"Processing TDI of cluster: {unit_data['cluster_id']}")
                cluster_id = unit_data['cluster_id']
                cluster_spikes = self.spike_times[self.cluster_id == cluster_id]
                spike_times = self.continuous_timestamps[cluster_spikes]
                
                # Calculate PSTH matrices for each condition (more efficient)
                condition_psth_matrices = {}
                condition_trial_counts = {}
                
                for cond in unique_conditions:
                    trial_mask = (conditions == cond) & good_trials
                    psth_matrix = self._calculate_psth_sliding_window(
                        spike_times, trial_mask, align_event, time_bins, window_size
                    )
                    if psth_matrix.size > 0:
                        condition_psth_matrices[cond] = psth_matrix
                        condition_trial_counts[cond] = psth_matrix.shape[0]
                
                # Calculate TDI for each time bin using your formula
                for time_idx in range(n_time_bins):
                    condition_rates = {}
                    condition_vars = {}
                    
                    for cond in condition_psth_matrices.keys():
                        rates_at_timebin = condition_psth_matrices[cond][:, time_idx]
                        condition_rates[cond] = np.mean(rates_at_timebin)
                        condition_vars[cond] = np.var(rates_at_timebin)
                    
                    # Calculate TDI using your formula
                    if len(condition_rates) >= 2:
                        rates_list = list(condition_rates.values())
                        r_max, r_min = np.nanmax(rates_list), np.nanmin(rates_list)
                        
                        # Calculate SSE (sum of squared errors)
                        sse = np.nansum([condition_vars[c] * condition_trial_counts[c] 
                                        for c in condition_rates.keys()])
                        
                        N = sum(condition_trial_counts.values())  # Total trials
                        M = len(condition_rates)  # Number of conditions
                        
                        # Check for valid TDI calculation - RELAXED THRESHOLDS
                        mean_fr = np.nanmean(rates_list)
                        if (not np.isnan(r_max) and not np.isnan(r_min) and 
                            (r_max - r_min) > 0 and N > M and sse >= 0 and mean_fr > 0.1):  
                            
                            tdi = (r_max - r_min) / (r_max - r_min + 2 * np.sqrt(sse / (N - M)))
                            tdi_matrix[unit_idx, time_idx] = tdi
                        else:
                            tdi_matrix[unit_idx, time_idx] = 0.0
                    else:
                        tdi_matrix[unit_idx, time_idx] = 0.0
                
                # Calculate baseline TDI for this unit
                baseline_tdi_values = tdi_matrix[unit_idx, baseline_indices]
                baseline_tdi = np.nanmean(baseline_tdi_values)

                tdi_baseline_corrected[unit_idx, :] = (tdi_matrix[unit_idx, :] - baseline_tdi) / (baseline_tdi + 0.0000001) 

                            
            except Exception as e:
                if self.verbose:
                    print(f"Error processing unit {unit_data['cluster_id']}: {e}")
        
        # Create single plot
        fig, ax = plt.subplots(1, 1, figsize=(12, max(6, n_units * 0.03)))
        vmin, vmax = 0, 2
        
        # FIXED: Correct extent using array dimensions
        extent = [time_start, time_end, 0, n_units]
        
        # Plot baseline-corrected TDI heatmap with viridis colormap
        im = ax.imshow(tdi_baseline_corrected, aspect='auto', cmap='viridis', 
                    interpolation='nearest', extent=extent, 
                    vmin=vmin, vmax=vmax, origin='lower')
        
        ax.set_title('TDI (Baseline Corrected)', fontweight='bold', fontsize=14)
        
        # FIXED: Proper y-ticks mapping array indices to real depths
        n_ticks = min(20, n_units)  # Limit number of ticks for readability
        if n_units > 1:
            tick_step = max(1, n_units // n_ticks)
            ytick_indices = list(range(0, n_units, tick_step))
            ytick_labels = []
            
            for i in ytick_indices:
                if i < len(valid_units):
                    ytick_labels.append(f"{valid_units[i]['depth']:.0f}")
            
            ax.set_yticks(ytick_indices)
            ax.set_yticklabels(ytick_labels)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.8, label='TDI (Baseline Corrected)')
        
        # Add labels and formatting
        ax.set_xlabel('Time from STIMON (s)', fontsize=12)
        ax.set_ylabel('Depth (μm)', fontsize=12)
        
        # Add stimulus period as shaded region
        ax.axvspan(0, 0.1, color='white', alpha=0.2)
        
        # Add baseline period marker
        ax.axvspan(baseline_start, baseline_end, color='gray', alpha=0.3, label='Baseline')
        
        # Add event markers
        stimulus_center_axis = (0.05 - time_start) / (time_end - time_start)
        go_cue_axis = (0.9 - time_start) / (time_end - time_start)
        
        ax.text(stimulus_center_axis, 1.0, 'Stimulus', transform=ax.transAxes, 
                ha='center', va='bottom', fontsize=10, color='black')
        ax.text(go_cue_axis, 1.0, 'Go Cue', transform=ax.transAxes, 
                ha='center', va='bottom', fontsize=10, color='black')
        
        # Add saccade line
        ax.axvline(0.9, color='red', linestyle=':', linewidth=2, alpha=0.8)
        
        # Add zero line for baseline reference (only if using subtraction baseline correction)
        # ax.axhline(y=n_units/2, color='white', linestyle='-', linewidth=1, alpha=0.5)
        
        print(f"Processed {n_units} units")
        print(f"Depth range: {valid_units[0]['depth']:.0f} - {valid_units[-1]['depth']:.0f} μm")
        print(f"Baseline period: {baseline_start} to {baseline_end} s")
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(self.base_dir, f"tdi_heatmap_baseline_corrected_sacc{self.sacc_file}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        self._print(f"TDI heatmap saved to: {save_path}")
        
        return tdi_baseline_corrected, valid_units, time_bins