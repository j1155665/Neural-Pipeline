# analysis_utils/data_loader.py

import numpy as np
import os
from pathlib import Path

class NeuralDataLoader:
    def __init__(self, data_dir=None):
        # Path to processed data directory. If None, uses default relative path.
        if data_dir is None:
            # Try to find the correct path automatically
            current_file = Path(__file__)
            
            # Check if we're in source/analysis_utils or just analysis_utils
            if 'source' in current_file.parts:
                self.data_dir = current_file.parent.parent.parent / "results" / "preprocessing"
            else:
                # We're in D:\Neural-Pipeline\analysis_utils\
                self.data_dir = current_file.parent.parent / "results" / "preprocessing"
                    
        else:
            self.data_dir = Path(data_dir)

          
        self.dots3DMP_data = None
        self.dots3DMPtuning_data = None
        self.subject = None
        self.date = None
        self.dots3DMP_good_trial = None
        self.dots3DMPtuning_good_trial = None
        
    def load_session(self, subject, date):
        # Load both dots3DMP and dots3DMPtuning data for a session
        self.subject = subject
        self.date = date
        
        # Load dots3DMP data
        dots3DMP_file = self.data_dir / f"{subject}{date}dots3DMP_processed.npz"
        if dots3DMP_file.exists():
            self.dots3DMP_data = np.load(dots3DMP_file, allow_pickle=True)
            print(f"Loaded dots3DMP data: {dots3DMP_file.name}")
            self.dots3DMP_good_trial = self.get_good_trials('dots3DMP')
        else:
            print(f"dots3DMP file not found: {dots3DMP_file}")
            
        # Load dots3DMPtuning data
        tuning_file = self.data_dir / f"{subject}{date}dots3DMPtuning_processed.npz"
        if tuning_file.exists():
            self.dots3DMPtuning_data = np.load(tuning_file, allow_pickle=True)
            print(f"Loaded dots3DMPtuning data: {tuning_file.name}")
            self.dots3DMPtuning_good_trial = self.get_good_trials('tuning')
        else:
            print(f"dots3DMPtuning file not found: {tuning_file}")
    
    def get_good_units(self):
        # Get indices of good units only
        if self.dots3DMP_data is None:
            print("No data loaded. Use load_session() first.")
            return None
        
        cluster_group = self.dots3DMP_data['cluster_group']
        good_units = np.where(cluster_group == 'good')[0]
        return good_units
    
    def get_good_trials(self, task='dots3DMP'):
        # Get the good trial indices for tuning and dots3DMP
        if task == 'dots3DMP':
            if self.dots3DMP_data is None:
                print("No dots3DMP data loaded.")
                return None
            # If already computed, return cached result
            if self.dots3DMP_good_trial is not None:
                return self.dots3DMP_good_trial
            # Otherwise compute and return
            goodtrial = self.dots3DMP_data['goodtrial']
            return np.where(goodtrial == 1)[0]
            
        elif task == 'tuning':
            if self.dots3DMPtuning_data is None:
                print("No dots3DMPtuning data loaded.")
                return None
            # If already computed, return cached result
            if self.dots3DMPtuning_good_trial is not None:
                return self.dots3DMPtuning_good_trial
            # Otherwise compute and return
            goodtrial = self.dots3DMPtuning_data['goodtrial']
            return np.where(goodtrial == 1)[0]
        else:
            print(f"Unknown task: {task}. Use 'dots3DMP' or 'tuning'.")
            return None
    def get_right_spkrate_config(self, spkrate):
        if spkrate.ndim == 4 and spkrate.shape[-1] == 1:
            spkrate = np.squeeze(spkrate, axis=-1)
        n_total_units = len(self.dots3DMP_data['cluster_group']) if self.dots3DMP_data is not None else 0
        n_total_trials = len(self.dots3DMP_data['goodtrial']) if self.dots3DMP_data is not None else 0
        
        dim0, dim1, dim2 = spkrate.shape
        
        # Find unit and event dimensions
        unit_dim = None
        event_dim = None
        
        if dim1 == n_total_units:
            unit_dim = 1
        elif dim0 == n_total_units:
            unit_dim = 0
        elif dim2 == n_total_units:
            unit_dim = 2
        
        if dim0 == n_total_trials:
            event_dim = 0
        elif dim1 == n_total_trials:
            event_dim = 1
        elif dim2 == n_total_trials:
            event_dim = 2
        
        # Default to [event, unit, timeaxis] if can't determine
        if unit_dim is None or event_dim is None:
            event_dim, unit_dim = 0, 1
        
        time_dim = ({0, 1, 2} - {unit_dim, event_dim}).pop()
        
        # Transpose to [unit, event, timeaxis]
        transpose_order = [unit_dim, event_dim, time_dim]
        spkrate = np.transpose(spkrate, transpose_order)
        
        return spkrate
    
    def get_spike_data(self, alignment='stimOn', good_units_only=True, good_trials_only=True):
        # Get spike rate data for specific alignment - returns [units, trials, time_bins]
        if self.dots3DMP_data is None:
            print("No dots3DMP data loaded.")
            return None
            
        alignment_map = {
            'stimOn': 'stimOn_spkrate',
            'saccOnset': 'saccOnset_spkrate', 
            'postTargHold': 'postTargHold_spkrate'
        }
        
        if alignment not in alignment_map:
            print(f"Unknown alignment: {alignment}")
            print(f"Available alignments: {list(alignment_map.keys())}")
            return None
            
        # Get spike rate data - shape: [event, unit, timeaxis, 1]
        spkrate = self.dots3DMP_data[alignment_map[alignment]]        
        spkrate = self.get_right_spkrate_config(spkrate)
        
        # Filter good units first
        if good_units_only:
            good_units = self.get_good_units()
            if good_units is not None:
                spkrate = spkrate[good_units]
        
        # Filter good trials - make sure indices are valid
        if good_trials_only:
            good_trials = self.get_good_trials('dots3DMP')
            if good_trials is not None:
                # Check bounds and filter invalid indices
                valid_trials = good_trials[good_trials < spkrate.shape[1]]
                if len(valid_trials) != len(good_trials):
                    print(f"Warning: {len(good_trials) - len(valid_trials)} trial indices out of bounds")
                spkrate = spkrate[:, valid_trials, :]
        
        return spkrate
    
    def get_tuning_data(self, good_units_only=True, good_trials_only=True):
        # Get tuning spike rate data - returns [units, trials, time_bins]
        if self.dots3DMPtuning_data is None:
            print("No dots3DMPtuning data loaded.")
            return None
            
        # Get spike rate data
        spkrate = self.dots3DMPtuning_data['spkrate']
        spkrate = self.get_right_spkrate_config(spkrate)
        
        # Filter good units
        if good_units_only:
            good_units = self.get_good_units()
            if good_units is not None:
                spkrate = spkrate[good_units]
        
        # Filter good trials
        if good_trials_only:
            good_trials = self.get_good_trials('tuning')
            if good_trials is not None:
                valid_trials = good_trials[good_trials < spkrate.shape[1]]
                if len(valid_trials) != len(good_trials):
                    print(f"Warning: {len(good_trials) - len(valid_trials)} trial indices out of bounds")
                spkrate = spkrate[:, valid_trials, :]
            
        return spkrate
    
    def get_behavioral_data(self, task='dots3DMP', good_trials_only=True):
        # Get behavioral data for analysis
        if task == 'dots3DMP' and self.dots3DMP_data is not None:
            behavioral_data = {}
            
            # Load and flatten behavioral variables
            keys = ['choice', 'PDW', 'modality', 'headingInd', 'coherenceInd', 
                    'goodtrial', 'deltaInd', 'correct', 'oneTargChoice', 'oneTargConf',
                    'heading', 'coherence', 'delta', 'RT']
            
            for key in keys:
                data = self.dots3DMP_data[key]
                # Flatten if it's a 2D array with one dimension being 1
                if data.ndim > 1:
                    data = np.squeeze(data)
                behavioral_data[key] = data
            
            # Filter good trials if requested
            if good_trials_only:
                good_trials = self.get_good_trials('dots3DMP')
                if good_trials is not None:
                    for key in behavioral_data:
                        behavioral_data[key] = behavioral_data[key][good_trials]
            
            return behavioral_data
            
        elif task == 'tuning' and self.dots3DMPtuning_data is not None:
            behavioral_data = {}
            
            # Load and flatten tuning behavioral variables
            keys = ['goodtrial', 'headingInd', 'modality', 'coherenceInd', 'deltaInd']
            
            for key in keys:
                data = self.dots3DMPtuning_data[key]
                # Flatten if it's a 2D array with one dimension being 1
                if data.ndim > 1:
                    data = np.squeeze(data)
                behavioral_data[key] = data
            
            # Filter good trials if requested
            if good_trials_only:
                good_trials = self.get_good_trials('tuning')
                if good_trials is not None:
                    for key in behavioral_data:
                        behavioral_data[key] = behavioral_data[key][good_trials]
            
            return behavioral_data
        else:
            print(f"No data available for task: {task}")
            return None
    
    def get_unit_info(self, good_units_only=False):
        # Get unit information
        if self.dots3DMP_data is None:
            print("No data loaded.")
            return None
        
        unit_info = {
            'depth': self.dots3DMP_data['depth'],
            'cluster_id': self.dots3DMP_data['cluster_id'],
            'cluster_group': self.dots3DMP_data['cluster_group']
        }
        
        if good_units_only:
            good_units = self.get_good_units()
            if good_units is not None:
                unit_info = {
                    'depth': unit_info['depth'][good_units],
                    'cluster_id': unit_info['cluster_id'][good_units], 
                    'cluster_group': unit_info['cluster_group'][good_units]
                }
        
        return unit_info

    def get_session_info(self):
        # Get information about the current session
        return {
            'subject': self.subject,
            'date': self.date,
            'n_units_total': len(self.dots3DMP_data['cluster_group']) if self.dots3DMP_data is not None else 0,
            'n_good_units': len(self.get_good_units()) if self.get_good_units() is not None else 0,
            'n_dots3DMP_trials': len(self.dots3DMP_data['goodtrial']) if self.dots3DMP_data is not None else 0,
            'n_dots3DMP_good_trials': len(self.get_good_trials('dots3DMP')) if self.get_good_trials('dots3DMP') is not None else 0,
            'n_tuning_trials': len(self.dots3DMPtuning_data['goodtrial']) if self.dots3DMPtuning_data is not None else 0,
            'n_tuning_good_trials': len(self.get_good_trials('tuning')) if self.get_good_trials('tuning') is not None else 0
        }
    
    def get_units_by_area(self, unit_info = None, area_name = 'dual'):
            """Get unit indices for specific brain area based on depth"""
            # get_unit_info is a method of NeuralDataLoader, not Dots3DMPConfig. So we need to pass unit_info as an argument.
            if unit_info is None:
                unit_info = self.get_unit_info(good_units_only=True)

            if self.date == "20250523":
                area_map = {
                    'MST': [0, 4000],
                    'VPS': [4001, 8000],
                    'MT': None,
                    'dual':[0, 8000]
                }
            elif self.date == "20250602":
                area_map = {
                    'MST': [0, 3500],
                    'VPS': [3500, 8000],
                    'MT': None,
                    'dual':[0, 8000]
                }
            elif self.date == "20250702":
                area_map = {
                    'MST': [1300, 7000],
                    'VPS': [7000, 8000],
                    'MT':  [0, 1300],
                    'dual':[0, 8000]
                }
            elif self.date == "20250710":
                area_map = {
                    'MST': [0, 4000],
                    'VPS': [4001, 8000],
                    'MT': None,
                    'dual':[0, 8000]
                }
            
            if area_name not in area_map:
                raise ValueError(f"Unknown area name: {area_name}")
            elif  area_map[area_name] is  None:
                valid_units = np.arange(len(unit_info['depth']))
            else:
                valid_units = np.where((unit_info['depth'] >= area_map[area_name][0]) & (unit_info['depth'] <= area_map[area_name][1]))[0]
        
            return valid_units
    

class Dots3DMPConfig:
    def __init__(self, subject):
        self.subject = subject
        self.event_info = {}
        self.tuning_event_info = {}
        self.time_info = {}
        self.tuning_time_info = {}
        
        self._setup_config()
    
    def _setup_config(self):
        if self.subject == 'zarya':
            self._setup_zarya_config()
        else:
            raise ValueError(f"Configuration not available for subject: {self.subject}")
    
    def _setup_zarya_config(self):
        # === Event Info (Main Analysis) ===
        self.event_info = {
            'name': ['modality', 'coherenceInd', 'headingInd', 'choice', 'PDW'],
            'class_1': [[], [], [1, 2, 3], 1, 0],
            'class_2': [[], [], [5, 6, 7], 2, 1],
            # Mapping dictionaries
            'modality_labels': {1: 'vestibular', 2: 'visual', 3: 'combined'},
            'coherence_values': {1: 0.2, 2: 0.7},
            'heading_values': {1: -10, 2: -3.9, 3: -1.5, 4: 0, 5: 1.5, 6: 3.9, 7: 10},
            'choice_labels': {1: 'left', 2: 'right'},
            'pdw_labels': {0: 'low bet', 1: 'high bet'},
            'stimulus_labels': {1: 'left', 2: 'right'}
        }
        
        # === Tuning Event Info ===
        self.tuning_event_info = {
            'name': ['modality', 'coherenceInd', 'headingInd'],
            'heading_values': [-45, -21.2, -10, -3.9, 3.9, 10, 21.2, 45],
            'class_1': [[], [], [1, 2, 3, 4]],
            'class_2': [[], [], [5, 6, 7, 8]],
            # Mapping for tuning
            'modality_labels': {1: 'vestibular', 2: 'visual', 3: 'combined'},
            'coherence_values': {1: 0.2, 2: 0.7},
            'heading_idx_to_value': {i+1: val for i, val in enumerate([-45, -21.2, -10, -3.9, 3.9, 10, 21.2, 45])}
        }
        
        # === Time Info (Main Analysis) ===
        self.time_info = {
            'offset': 0.05,
            'bin_size': 0.02,
            'align_events': ['stimOn', 'saccOnset', 'postTargHold'],
            'plot_names': ['Stim On', 'Choice', 'PDW'],
            'center_start': [-0.1, -0.6, -0.4],
            'center_stop': [0.8, 0.3, 0.5],
            'sigma': 0,
            'vel_profile_dt': 0.0083,
            'max_velocity': 0.66,
            'max_acceleration': 0.4,
            'max_deceleration': 0.93,
        }
        
        # === Time Info (Tuning Analysis) ===
        self.tuning_time_info = {
            'offset': 0.05,
            'bin_size': 0.02,
            'align_events': ['stimOn'],
            'plot_names': ['Stim On'],
            'center_start': [-0.1],
            'center_stop': [2.2],
            'sigma': 0,
            'vel_profile_dt': 0.0083,
            'trial_start': -0.2,
            'trial_stop': 0.2
        }
        
        print(f'Loaded dots3DMP configuration for subject: {self.subject}')
    
    # === Conversion Methods ===
    def get_modality_label(self, modality_idx):
        """Convert modality index to label"""
        return self.event_info['modality_labels'].get(modality_idx, f'unknown_{modality_idx}')
    
    def get_coherence_value(self, coherence_idx):
        """Convert coherence index to actual value"""
        return self.event_info['coherence_values'].get(coherence_idx, f'unknown_{coherence_idx}')
    
    def get_heading_value(self, heading_idx, task='dots3DMP'):
        """Convert heading index to actual angle value"""
        if task == 'dots3DMP':
            return self.event_info['heading_values'].get(heading_idx, f'unknown_{heading_idx}')
        elif task == 'tuning':
            return self.tuning_event_info['heading_idx_to_value'].get(heading_idx, f'unknown_{heading_idx}')
    
    def get_choice_label(self, choice_idx):
        """Convert choice index to label"""
        return self.event_info['choice_labels'].get(choice_idx, f'unknown_{choice_idx}')
    
    def get_pdw_label(self, pdw_idx):
        """Convert PDW index to label"""
        return self.event_info['pdw_labels'].get(pdw_idx, f'unknown_{pdw_idx}')
    
    def convert_behavioral_data(self, behavioral_data, task='dots3DMP'):
        """Add converted/labeled versions of behavioral data"""
        converted = behavioral_data.copy()
        
        if 'modality' in behavioral_data:
            converted['modality_labels'] = np.array([self.get_modality_label(m) for m in behavioral_data['modality']])
        
        if 'coherenceInd' in behavioral_data:
            converted['coherence_values'] = np.array([self.get_coherence_value(c) for c in behavioral_data['coherenceInd']])
        
        if 'headingInd' in behavioral_data:
            converted['heading_values'] = np.array([self.get_heading_value(h, task) for h in behavioral_data['headingInd']])
        
        if 'choice' in behavioral_data:
            converted['choice_labels'] = np.array([self.get_choice_label(c) for c in behavioral_data['choice']])
        
        if 'PDW' in behavioral_data:
            converted['pdw_labels'] = np.array([self.get_pdw_label(p) for p in behavioral_data['PDW']])
        
        return converted
    
    def get_time_Info(self, task='dots3DMP'):
        """Generate time axis for plotting"""
        if task == 'dots3DMP':
            return self.time_info
        else:  # tuning
             return self.tuning_time_info
        
    def get_time_axes(self, task='dots3DMP'):
        """Generate time axes for all alignments"""
        time_info = self.get_time_Info(task)
        
        time_axes = {}
        align_events = time_info['align_events']
        center_start = time_info['center_start']
        center_stop = time_info['center_stop']
        bin_size = time_info['bin_size']
        
        for i, alignment in enumerate(align_events):
            start_time = center_start[i]
            stop_time = center_stop[i]
            time_axis = np.arange(start_time, stop_time + bin_size, bin_size)
            time_axes[alignment] = time_axis
        
        return time_axes
    
    def get_class_trials(self, behavioral_data, variable, class_num, task='dots3DMP'):
        """Get trial indices for specific class"""
        if task == 'dots3DMP':
            event_info = self.event_info
        else:
            event_info = self.tuning_event_info
        
        var_idx = event_info['name'].index(variable)
        class_values = event_info[f'class_{class_num}'][var_idx]
        
        if len(class_values) == 0:
            return np.arange(len(behavioral_data[variable]))  # all trials
        
        return np.isin(behavioral_data[variable], class_values)
    
 
    