import os
import re
import numpy as np
import pandas as pd
import scipy.io
from scipy.io import loadmat, savemat
import types 
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

class MergeRecordingFile:
    def __init__(self, directory, subject, date):
        self.directory = directory
        self.subject = subject
        self.date = date
        self.filedate = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        self.pattern = re.compile(f"{self.filedate}_\\d{{2}}-\\d{{2}}-\\d{{2}}")

        self.matching_files = self.find_matching_files()
        self.matching_data = self.find_matching_data()
        self.time_adjustment = self.check_time_adjustment()
        self.ttl_unique = self.check_unique_block()

    def find_matching_files(self):
        """Finds recording session folders based on date pattern."""
        files_path = os.path.join(self.directory, f"{self.date}")
        files = os.listdir(files_path)
        matched_files = [f for f in files if self.pattern.match(f)]
        if not matched_files:
            raise ValueError(f"No files found for subject {self.subject} on {self.date}.")
        return matched_files

    def find_matching_data(self):
        """Finds the paths to recording data inside matched session folders, sorted by time, experiment, then recording."""
        matching_data = []
        for file in self.matching_files:
            data_directory = os.path.join(self.directory, self.date, file, "Record Node 101")
            if not os.path.exists(data_directory):
                continue
            for root, dirs, files in os.walk(data_directory):
                path_parts = root.split(os.sep)
                if len(path_parts) >= 2 and "recording" in path_parts[-1].lower():
                    matching_data.append(root)
        
        def sort_key(path):
            parts = path.split(os.sep)
            session_part = next((p for p in parts if re.match(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', p)), '')
            experiment_num = next((int(p.replace('experiment', '')) for p in parts if p.startswith('experiment')), 0)
            recording_num = next((int(p.replace('recording', '')) for p in parts if p.startswith('recording')), 0)
            return (session_part, experiment_num, recording_num)
        
        matching_data.sort(key=sort_key)
        return matching_data

    def get_electrode_configuration(self, filepath, attribute="electrodeConfigurationPreset"):
        """Extracts the electrode configuration from settings.xml."""
        settings_path = os.path.join(filepath, 'settings.xml')
        if not os.path.exists(settings_path):
            return None
        try:
            tree = ET.parse(settings_path)
            root = tree.getroot()
            for elem in root.iter():
                if attribute in elem.attrib:
                    return elem.attrib[attribute]
        except ET.ParseError:
            return None
        return None
    
    def extract_channel_positions(self, AP_name='ProbeA'):
        """Get the configuration of the probe as ProbeConfig.txt"""

        xml_file = os.path.join(self.directory, self.date, self.matching_files[0], 'Record Node 101', 'settings.xml')
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Find all NP_PROBE nodes
        probe_nodes = root.findall(".//NP_PROBE")
        
        probe_node = None
        for node in probe_nodes:

            probe_serial = node.get('probe_serial_number')
            custom_name = node.get('custom_probe_name')
            
            # For now, we'll match by order: ProbeA = first probe, ProbeB = second probe
            # You can modify this logic based on your needs
            if AP_name == 'ProbeA' and node == probe_nodes[0]:
                probe_node = node
                break
            elif AP_name == 'ProbeB' and node == probe_nodes[1]:
                probe_node = node
                break
        
        if probe_node is None:
            print(f"Probe {AP_name} not found in XML")
            return

        # Extract electrode positions
        xpos_node = probe_node.find("ELECTRODE_XPOS")
        ypos_node = probe_node.find("ELECTRODE_YPOS")

        xpos = {int(k[2:]): int(v) for k, v in xpos_node.attrib.items() if k.startswith("CH")}
        ypos = {int(k[2:]): int(v) for k, v in ypos_node.attrib.items() if k.startswith("CH")}

        ch_positions = [(ch, xpos.get(ch, None), ypos.get(ch, None)) for ch in sorted(set(xpos) | set(ypos))]

        x_pos = []
        y_pos = []
        for ch, x, y in ch_positions:
            x_pos.append(x)
            y_pos.append(y)

        # Save with probe name in filename
        file_path = os.path.join(self.directory, self.date, f"ProbeConfig_{AP_name}.txt")
        with open(file_path, 'w') as f:
            f.write(f"Probe: {AP_name}\n")
            f.write(f"Serial Number: {probe_node.get('probe_serial_number')}\n")
            f.write(f"Part Number: {probe_node.get('probe_part_number')}\n\n")
            f.write("X-pos\n")
            f.write(f"[{', '.join(str(x) for x in x_pos)}]\n\n")
            f.write("Y-pos\n")
            f.write(f"[{', '.join(str(y) for y in y_pos)}]\n\n")

        print(f"Channel configuration for {AP_name} saved to '{file_path}'.")

        return

    def check_electrode_consistency(self):
        """Checks if electrode configuration is consistent across all recordings."""
        consistent_preset = None
        for file in self.matching_files:
            recording_dir = os.path.join(self.directory, file, 'Record Node 101')
            preset = self.get_electrode_configuration(recording_dir)
            if preset is None:
                continue
            if consistent_preset is None:
                consistent_preset = preset
            elif consistent_preset != preset:
                raise ValueError(f"Inconsistent electrodeConfigurationPreset in {file}: {preset} (expected {consistent_preset}).")
        return consistent_preset
    
    def check_time_adjustment(self):
        """Check if we need to adjust the timestamp across trials."""
        start_time = []
        stop_time = []
        delta_time_sum = 0
        delta_time = []
        time_adjustment = []
    
        for i, file in enumerate(self.matching_data):
            # Load AP timestamps and add block info
            apt_filepath = os.path.join(file, 'continuous', 'Neuropix-PXI-100.ProbeA-AP', 'timestamps.npy')
            if os.path.exists(apt_filepath):

                ap_timestamps = np.load(apt_filepath)
                start_time.append(ap_timestamps[0])
                stop_time.append(ap_timestamps[-1])
                delta_time.append(ap_timestamps[-1] - ap_timestamps[0] + delta_time_sum)
                delta_time_sum += (ap_timestamps[-1] - ap_timestamps[0])
            else:
                print(f"Warning: AP timestamps file not found at {file}")

        time_adjustment.append(-start_time[0])
        for i, diff in enumerate(delta_time[0:-1]):
            time_adjustment.append(-start_time[i+1] + diff + 1)

        return time_adjustment
    
    def check_unique_block(self):
        """Check the unique number (time) for the file name"""

        ttl_unique_list = []
        for i, file in enumerate(self.matching_data):


            base_path = os.path.join(file, 'events')
            daq_folder = None
            for folder in os.listdir(base_path):
                if folder.startswith('NI-DAQmx-') and folder.endswith('.PXIe-6341'):
                    daq_folder = os.path.join(base_path, folder)
                    break

            if daq_folder is None:
                raise FileNotFoundError("No valid NI-DAQmx PXIe-6341 folder found.")

            ttl_filepath = os.path.join(daq_folder, 'TTL', 'full_words.npy')

            # Load TTL data and add block markers
            if os.path.exists(ttl_filepath):

                ttl_data = np.load(ttl_filepath)

                ttl_data[ttl_data >= 256] -= 256
                ttl_data = ttl_data[(ttl_data > 0) & (ttl_data < 256)]
                unique_ttl = np.unique(ttl_data, return_index=True)
                ordered_unique = unique_ttl[0][np.argsort(unique_ttl[1])]
                
                if ordered_unique.size > 3:
                    ones_digit = ordered_unique[2]
                    
                    if ones_digit == 2 and ordered_unique[3] == 3:
                        ones_digit = 0
                    elif ones_digit == 2 and ordered_unique[3] == 2:
                        ones_digit = 2
                    
                    ttl_unique = ordered_unique[1] * 100 + ones_digit
                    ttl_unique_list.append(ttl_unique)
 
                else:
                    print(f"Warning: Not enough unique values for file: {ttl_filepath}")

            else:
                print(f"Warning: TTL file not found at {file}")

        print("valid PDS filetime:", [int(x) for x in ttl_unique_list])

        return ttl_unique_list

    def merge_ap_data(self, num_channels=384, AP_name = 'ProbeA'): # after 20250718, upgrade openephys 1.0, num_channel == 384, before that, num_channel == 385
        """Merges AP (action potential) data from multiple recording files and includes block information."""
        if AP_name in ['ProbeA']:
            output_path = os.path.join(self.directory, f"{self.date}", f"{self.subject}{self.date}dots3DMP_AP.dat")
            aptime_out_path = os.path.join(self.directory, f"{self.date}", f"{self.subject}{self.date}dots3DMP_APtimestamps.npy")
            aptime_block_path = os.path.join(self.directory, f"{self.date}", f"{self.subject}{self.date}dots3DMP_APblocks.npy")       
        else:
            output_path = os.path.join(self.directory, f"{self.date}", f"{self.subject}{self.date}dots3DMP_AP{AP_name}.dat")
            aptime_out_path = os.path.join(self.directory, f"{self.date}", f"{self.subject}{self.date}dots3DMP_AP{AP_name}timestamps.npy")
            aptime_block_path = os.path.join(self.directory, f"{self.date}", f"{self.subject}{self.date}dots3DMP_AP{AP_name}blocks.npy")

        # Lists to store AP timestamps and block markers
        ap_timestamps_list = []
        ap_block_list = []
        
        final_original_size = 0

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        for i, file in enumerate(self.matching_data):
            print(f"Processing file {i+1}/{len(self.matching_data)}: {file}")
            
            # Path to continuous data
            filepath = os.path.join(file, 'continuous', f"Neuropix-PXI-100.{AP_name}-AP", 'continuous.dat')

            # Check if the file exists
            if os.path.exists(filepath):
                size_in_gb = os.path.getsize(filepath) / (1024 ** 3)
            else:
                print(f"File not found at {filepath}")
                continue 
            
            # Load data using memmap
            data = np.memmap(filepath, dtype='int16')
            data_size = data.size * 2 / (1024 ** 3)  # Size in GB
            
            # Check if the loaded data size matches the file size
            if abs(size_in_gb - data_size) < 1e-6:  
                print(f"Loaded data size matches: {data_size:.2f} GB")
            else:
                print(f"Size mismatch: {size_in_gb:.2f} GB (file) vs {data_size:.2f} GB (loaded). Stopping merging.")
                break
            
            # Accumulate total original size
            final_original_size += data_size
            
            # Reshape the data
            data = np.reshape(data, (data.size // num_channels, num_channels))

            # Determine the mode for file opening (write or append)
            mode = 'wb' if i == 0 else 'ab'
            with open(output_path, mode) as f:
                data.tofile(f)  # Write or append data
                end_size = os.path.getsize(output_path)
                print(f"Merged file size: {end_size / (1024 ** 3):.2f} GB")

            # Load AP timestamps and add block info
            apt_filepath = os.path.join(file, 'continuous', f"Neuropix-PXI-100.{AP_name}-AP", 'timestamps.npy')
            if os.path.exists(apt_filepath):
                ap_timestamps = np.load(apt_filepath)
                ap_timestamps += np.ones(ap_timestamps.shape) * self.time_adjustment[i]
                ap_timestamps_list.append(ap_timestamps)

                ap_block = np.ones(ap_timestamps.shape) * self.ttl_unique[i]  # Add block number
                ap_block_list.append(ap_block)
            else:
                print(f"Warning: AP timestamps file not found at {file}")
        
        # Merge AP timestamps and save
        if ap_timestamps_list:
            merged_ap_timestamps = np.concatenate(ap_timestamps_list)
            np.save(aptime_out_path, merged_ap_timestamps)
            np.save(aptime_block_path, np.concatenate(ap_block_list))
            print(f"Merged AP timestamps saved")
        else:
            print("No AP timestamps to merge.")  
        
        # Final file size and size verification
        final_size = os.path.getsize(output_path) / (1024 ** 3)  # Size in GB
        print(f"Final merged file size: {final_size:.2f} GB")
        
        if abs(final_size - final_original_size) < 1e-6:  
            print("Final size matches the total original size.")
        else:
            print(f"WARNING: Size mismatch! Difference: {final_size - final_original_size:.2f} GB")

        return 


    def merge_ttl_data(self):
        """Merges TTL data and timestamps from multiple recordings, adds block markers, and saves the merged data."""
        ttl_out_path = os.path.join(self.directory, f"{self.date}", f"{self.subject}{self.date}dots3DMP_ttl.npy")
        ttl_block_path = os.path.join(self.directory, f"{self.date}", f"{self.subject}{self.date}dots3DMP_ttlblocks.npy")
        ttltime_out_path = os.path.join(self.directory, f"{self.date}", f"{self.subject}{self.date}dots3DMP_ttltimestamps.npy")

        ttl_list = []
        ttl_timestamps_list = []
        ttl_block_list = []

        os.makedirs(os.path.dirname(ttl_out_path), exist_ok=True)

        for i, file in enumerate(self.matching_data):

            base_path = os.path.join(file, 'events')
            daq_folder = None
            for folder in os.listdir(base_path):
                if folder.startswith('NI-DAQmx-') and folder.endswith('.PXIe-6341'):
                    daq_folder = os.path.join(base_path, folder)
                    break

            if daq_folder is None:
                raise FileNotFoundError("No valid NI-DAQmx PXIe-6341 folder found.")

            ttl_filepath = os.path.join(daq_folder, 'TTL', 'full_words.npy')
            ttlt_filepath = os.path.join(daq_folder, 'TTL', 'timestamps.npy')

            # Load TTL data and add block markers
            if os.path.exists(ttl_filepath):

                ttl_data = np.load(ttl_filepath)
                ttl_list.append(ttl_data)

                ttl_block = np.ones(ttl_data.shape)
                ttl_block *= self.ttl_unique[i]  # Add block number for each file
                ttl_block_list.append(ttl_block)

            else:
                print(f"Warning: TTL file not found at {file}")

            # Load TTL timestamps
            if os.path.exists(ttlt_filepath):
                ttl_timestamps = np.load(ttlt_filepath)
                ttl_timestamps += np.ones(ttl_timestamps.shape) * self.time_adjustment[i]
                ttl_timestamps_list.append(ttl_timestamps)
            else:
                print(f"Warning: TTL timestamps file not found at {file}")

        # Merge TTL data and save
        if ttl_list:
            merged_ttl = np.concatenate(ttl_list)
            np.save(ttl_out_path, merged_ttl)
            np.save(ttl_block_path, np.concatenate(ttl_block_list))
            print(f"Merged TTL saved")
        else:
            print("No TTL data to merge.")

        # Merge TTL timestamps and save
        if ttl_timestamps_list:
            merged_ttl_timestamps = np.concatenate(ttl_timestamps_list)
            np.save(ttltime_out_path, merged_ttl_timestamps)
            print(f"Merged TTL timestamps saved")
        else:
            print("No TTL timestamps to merge.")

        return 

    def merge_eye_data(self,num_channels = 4):
        """Merges eye data from multiple recording files and includes block information."""
        output_path = os.path.join(self.directory, f"{self.date}", f"{self.subject}{self.date}dots3DMP_eyeXY.dat")
        eyetime_out_path = os.path.join(self.directory, f"{self.date}", f"{self.subject}{self.date}dots3DMP_eyeXYtimestamps.npy")
        eyetime_block_path = os.path.join(self.directory, f"{self.date}", f"{self.subject}{self.date}dots3DMP_eyeXYblocks.npy")

        # Lists to store AP timestamps and block markers
        eye_timestamps_list = []
        eye_block_list = []

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        for i, file in enumerate(self.matching_data):
            print(f"Processing eyeXY file {i+1}/{len(self.matching_data)}: {file}")
            
            # Path to continuous data
            base_path = os.path.join(file, 'continuous')
            daq_folder = None
            for folder in os.listdir(base_path):
                if folder.startswith('NI-DAQmx-') and folder.endswith('.PXIe-6341'):
                    daq_folder = os.path.join(base_path, folder)
                    break

            if daq_folder is None:
                raise FileNotFoundError("No valid NI-DAQmx PXIe-6341 folder found.")

            filepath = os.path.join(daq_folder, 'continuous.dat')

            # Check if the file exists
            if not os.path.exists(filepath):
                print(f"File not found at {filepath}")
                continue 
            
            # Load data using memmap
            data = np.memmap(filepath, dtype='int16')
            
            # Reshape the data
            data = np.reshape(data, (data.size // num_channels, num_channels))

            # Determine the mode for file opening (write or append)
            mode = 'wb' if i == 0 else 'ab'
            with open(output_path, mode) as f:
                data.tofile(f)  # Write or append data

            # Load eyeXY timestamps and add block info
            eyet_filepath = os.path.join(daq_folder, 'timestamps.npy')
            if os.path.exists(eyet_filepath):
                eye_timestamps = np.load(eyet_filepath)
                eye_timestamps += np.ones(eye_timestamps.shape) * self.time_adjustment[i]
                eye_timestamps_list.append(eye_timestamps)

                eye_block = np.ones(eye_timestamps.shape) * self.ttl_unique[i]  # Add block number
                eye_block_list.append(eye_block)
            else:
                print(f"Warning: eyeXY timestamps file not found at {file}")
        
        # Merge AP timestamps and save
        if eye_timestamps_list:
            merged_eye_timestamps = np.concatenate(eye_timestamps_list)
            np.save(eyetime_out_path, merged_eye_timestamps)
            np.save(eyetime_block_path, np.concatenate(eye_block_list))
            print(f"Merged eyeXY timestamps saved")
        else:
            print("No eyeXY timestamps to merge.")  

        return 


class CreateEventStruct:

    def __init__(self, directory, subject, date, AP_name='ProbeA'):
        self.directory = directory
        self.subject = subject
        self.date = date
        self.data_path = os.path.join(directory, date)
        self.probe_name = AP_name
        
        self.full_words = np.load(os.path.join(self.data_path, f"{subject}{date}dots3DMP_ttl.npy"))
        self.timestamps = np.load(os.path.join(self.data_path, f"{subject}{date}dots3DMP_ttltimestamps.npy"))
        self.ttl_blocks = np.load(os.path.join(self.data_path, f"{subject}{date}dots3DMP_ttlblocks.npy"))
        
        self.full_words[self.full_words >= 256] -= 256
        self.block_indices = self.ttl_blocks >= 0  # Include all blocks
        self.block_type = ['dots3DMP', 'dots3DMPtuning', 'dots3DMP_fixedRT', 'SaccadeTraining']
        self.event_data = None
        
        # Saccade training target locations
        self.sacc_angles = np.array([0, 0, 90, 90, 180, 180, 270, 270, 
                                     42, 42, 42, 138, 138, 138, 
                                     211, 211, 211, 329, 329, 329])
        self.sacc_eccs = np.array([5, 10, 5, 10, 5, 10, 5, 10, 
                                   3.5, 6.72, 10, 3.5, 6.72, 10, 
                                   3.5, 5.83, 10, 3.5, 5.83, 10])

    def filter_events(self):
        """ Filters event timestamps and words based on valid indices. """
        valid_indices = (self.full_words > 0) & (self.full_words != 13) & (self.full_words <= 256)

        if np.any(self.block_indices):
            self.filtered_full_words = self.full_words[valid_indices & self.block_indices]
            self.filtered_timestamps = self.timestamps[valid_indices & self.block_indices]
            self.filtered_ttl_blocks = self.ttl_blocks[valid_indices & self.block_indices]
        else:
            self.filtered_full_words = self.full_words[valid_indices]
            self.filtered_timestamps = self.timestamps[valid_indices]
            self.filtered_ttl_blocks = self.ttl_blocks[valid_indices]

    def load_info_data(self):
        """ Loads MATLAB `.mat` file and extracts necessary parameters. """
        mat_data = scipy.io.loadmat(os.path.join(self.data_path, f"{self.subject}{self.date}dots3DMP_info.mat"))
        info = mat_data['info']

        self.pldaps_filetimes = info['pldaps_filetimes'][0][0].flatten()
        self.par = info['par'][0, 0]

    def process_saccade_trials(self, block_indices):
        """Process SaccadeTraining trials with specific event structure"""
        data_block = self.filtered_full_words[block_indices]
        time_block = self.filtered_timestamps[block_indices]
        block_nums = self.filtered_ttl_blocks[block_indices]
        
        # Event codes for saccade training
        TRIAL, FIX, FIXATION, STIMON, STIMOFF, SACC, REWARD, BREAKFIX = 1, 2, 3, 4, 4, 6, 9, 10
        
        # Find trial indices - use ALL trial events, not filtered ones
        trial_indices_local = np.where(data_block == TRIAL)[0]
        fix_indices = np.where(data_block == FIX)[0]
        

        if trial_indices_local.size == 0:
            return None
        
        # This is the key difference - your original code processes ALL trial_indices
        # not just the filtered ones
        n_trials = len(trial_indices_local)
        
        # Initialize event structure for saccade training
        sacc_events = {
            'fpOn': np.full(n_trials, np.nan),
            'fixation': np.full(n_trials, np.nan),
            'stimOn': np.full(n_trials, np.nan),
            'stimOff': np.full(n_trials, np.nan),
            'saccOnset': np.full(n_trials, np.nan),
            'reward': np.full(n_trials, np.nan),
            'breakFix': np.full(n_trials, np.nan),
            'goodtrial': np.full(n_trials, np.nan),
            'condition': np.full(n_trials, np.nan),
            'target_angle': np.full(n_trials, np.nan),
            'target_ecc': np.full(n_trials, np.nan),
            'block': np.full(n_trials, np.nan)
        }
        
        # Process each trial - using the original trial_indices_local
        for i, trial_idx in enumerate(trial_indices_local):
            # Define trial boundaries like your original code
            trial_start = trial_indices_local[i-1] if i != 0 else (fix_indices[0] if fix_indices.size > 0 else 0)
            trial_end = trial_idx  # Current trial marker
            
            # Extract trial segment
            trial_data = data_block[trial_start:trial_end]
            trial_times = time_block[trial_start:trial_end]
            
            # Extract event timestamps - matching your original logic exactly
            # Use matching[-1] for STIMOFF, matching[0] for others
            fpOn_idx = np.where(trial_data == FIX)[0]
            sacc_events['fpOn'][i] = trial_times[fpOn_idx[0]] if fpOn_idx.size > 0 else np.nan
            
            fixation_idx = np.where(trial_data == FIXATION)[0]
            sacc_events['fixation'][i] = trial_times[fixation_idx[0]] if fixation_idx.size > 0 else np.nan
            
            stimOn_idx = np.where(trial_data == STIMON)[0]
            sacc_events['stimOn'][i] = trial_times[stimOn_idx[0]] if stimOn_idx.size > 0 else np.nan
            
            # STIMOFF uses last occurrence (matching[-1])
            stimOff_idx = np.where(trial_data == STIMOFF)[0]
            sacc_events['stimOff'][i] = trial_times[stimOff_idx[-1]] if stimOff_idx.size > 0 else np.nan
            
            sacc_idx = np.where(trial_data == SACC)[0]
            sacc_events['saccOnset'][i] = trial_times[sacc_idx[0]] if sacc_idx.size > 0 else np.nan
            
            reward_idx = np.where(trial_data == REWARD)[0]
            sacc_events['reward'][i] = trial_times[reward_idx[0]] if reward_idx.size > 0 else np.nan
            
            breakfix_idx = np.where(trial_data == BREAKFIX)[0]
            sacc_events['breakFix'][i] = trial_times[breakfix_idx[0]] if breakfix_idx.size > 0 else np.nan
            
            # Extract condition code (target location)
            # Your original: trial_ttls > 100
            cond_idx = np.where(trial_data > 100)[0]
            if cond_idx.size > 0:
                cond_code = trial_data[cond_idx[0]] - 100
                sacc_events['condition'][i] = cond_code
                # Map condition to angle and eccentricity
                if 0 <= cond_code < len(self.sacc_angles):
                    sacc_events['target_angle'][i] = self.sacc_angles[int(cond_code)]
                    sacc_events['target_ecc'][i] = self.sacc_eccs[int(cond_code)]
            else:
                sacc_events['condition'][i] = np.nan
            
            # Mark good trials - matching your original logic exactly
            sacc_events['goodtrial'][i] = 0 if not np.isnan(sacc_events['breakFix'][i]) else 1
            
            # Store block info
            sacc_events['block'][i] = block_nums[trial_idx]
        
        return sacc_events

    def process_dots3DMP_trials(self, block_indices, task_type):
        """Process dots3DMP, dots3DMPtuning, and dots3DMP_fixedRT trials"""
        data_block = self.filtered_full_words[block_indices]
        time_block = self.filtered_timestamps[block_indices]
        block_nums = self.filtered_ttl_blocks[block_indices]
        
        # Event codes
        TRIAL, FIX, FIXATION, STIMONOFF, SACC, TARGHOLD, POSTTARGHOLD, REWARD, BREAKFIX = 1, 2, 3, 5, 6, 7, 8, 9, 10
        
        # Find trial starts
        print(data_block[:3000])  # Debug: print first 3000 events to check for TRIAL code
        trial_indices_local = np.where(data_block == TRIAL)[0]
        
        if trial_indices_local.size == 0:
            return None
            
        idx_diffs = np.diff(trial_indices_local) > 10
        valid_trials = np.concatenate(([trial_indices_local[0]], trial_indices_local[1:][idx_diffs]))
        
        n_trials = len(valid_trials)
        
        # Initialize event structure
        mp_events = {
            'fpOn': np.full(n_trials, np.nan),
            'fixation': np.full(n_trials, np.nan),
            'stimOn': np.full(n_trials, np.nan),
            'stimOff': np.full(n_trials, np.nan),
            'saccOnset': np.full(n_trials, np.nan),
            'targHold': np.full(n_trials, np.nan),
            'postTargHold': np.full(n_trials, np.nan),
            'reward': np.full(n_trials, np.nan),
            'breakFix': np.full(n_trials, np.nan),
            'goodtrial': np.full(n_trials, np.nan),
            'headingInd': np.full(n_trials, np.nan),
            'modality': np.full(n_trials, np.nan),
            'coherenceInd': np.full(n_trials, np.nan),
            'deltaInd': np.full(n_trials, np.nan),
            'choice': np.full(n_trials, np.nan),
            'correct': np.full(n_trials, np.nan),
            'PDW': np.full(n_trials, np.nan),
            'block': np.full(n_trials, np.nan)
        }
        
        # Get event indices for the block
        fpOn_times = time_block[data_block == FIX]
        fixation_times = time_block[data_block == FIXATION]
        stimOnOff_times = time_block[data_block == STIMONOFF]
        sacc_times = time_block[data_block == SACC]
        targHold_times = time_block[data_block == TARGHOLD]
        postTargHold_times = time_block[data_block == POSTTARGHOLD]
        reward_times = time_block[data_block == REWARD]
        breakFix_times = time_block[data_block == BREAKFIX]
        
        vars_to_process = [
            ('deltaInd', 'PDW'),
            ('coherenceInd', 'correct'),
            ('modality', 'choice'),
            ('headingInd',)
        ]
        
        # Process each trial
        for j, trial_idx in enumerate(valid_trials):
            current_trial = time_block[trial_idx]
            previous_trial = time_block[0] if j == 0 else time_block[valid_trials[j - 1]]
            
            # Extract fpOn
            fpOn_valid = fpOn_times[(fpOn_times > previous_trial) & (fpOn_times < current_trial)]
            mp_events['fpOn'][j] = fpOn_valid[0] if fpOn_valid.size > 0 else np.nan
            
            refined_start = mp_events['fpOn'][j] if not np.isnan(mp_events['fpOn'][j]) else previous_trial
            
            # Extract other events
            fix_valid = fixation_times[(fixation_times > refined_start) & (fixation_times < current_trial)]
            mp_events['fixation'][j] = fix_valid[0] if fix_valid.size > 0 else np.nan
            
            stimOnOff_valid = stimOnOff_times[(stimOnOff_times > refined_start) & (stimOnOff_times < current_trial)]
            mp_events['stimOn'][j] = stimOnOff_valid[0] if stimOnOff_valid.size > 0 else np.nan
            mp_events['stimOff'][j] = stimOnOff_valid[-1] if stimOnOff_valid.size > 0 else np.nan
            
            sacc_valid = sacc_times[(sacc_times > refined_start) & (sacc_times < current_trial)]
            mp_events['saccOnset'][j] = sacc_valid[0] if sacc_valid.size > 0 else np.nan
            
            targHold_valid = targHold_times[(targHold_times > refined_start) & (targHold_times < current_trial)]
            mp_events['targHold'][j] = targHold_valid[0] if targHold_valid.size > 0 else np.nan
            
            postTargHold_valid = postTargHold_times[(postTargHold_times > refined_start) & (postTargHold_times < current_trial)]
            mp_events['postTargHold'][j] = postTargHold_valid[0] if postTargHold_valid.size > 0 else np.nan
            
            reward_valid = reward_times[(reward_times > refined_start) & (reward_times < current_trial)]
            mp_events['reward'][j] = reward_valid[0] if reward_valid.size > 0 else np.nan
            
            breakFix_valid = breakFix_times[(breakFix_times > refined_start) & (breakFix_times < current_trial)]
            mp_events['breakFix'][j] = breakFix_valid[0] if breakFix_valid.size > 0 else np.nan
            
            # Store block info
            mp_events['block'][j] = block_nums[trial_idx]
            
            # Check if trial is good
            breakfix = mp_events['breakFix'][j]
            required_events = ['fixation', 'stimOn', 'stimOff']
            
            if np.isnan(breakfix) and all(not np.isnan(mp_events[event][j]) for event in required_events):
                mp_events['goodtrial'][j] = 1
            else:
                mp_events['goodtrial'][j] = 0
                for var_tuple in vars_to_process:
                    for var in var_tuple:
                        mp_events[var][j] = np.nan
                continue
            
            # Extract behavioral variables
            if j == n_trials - 1:
                event_info = data_block[trial_idx:]
            else:
                event_info = data_block[trial_idx:valid_trials[j + 1]]
            
            event_info = event_info.copy() - 70
            
            for idx, var_tuple in enumerate(vars_to_process):
                var1 = var_tuple[0]
                var2 = var_tuple[1] if len(var_tuple) > 1 else None
                
                if idx > 0:
                    event_info -= 10
                    
                if "headingInd" in var1:
                    new_idx = (event_info <= 20) & (event_info >= 0)
                else:
                    new_idx = (event_info <= 10) & (event_info >= 0)
                
                new_info = event_info[new_idx]
                
                if new_info.size > 0:
                    mp_events[var1][j] = new_info[0]
                    if var2:
                        mp_events[var2][j] = new_info[-1]
                        if mp_events['breakFix'][j] == 0:
                            mp_events[var2][j] = np.nan
                else:
                    mp_events[var1][j] = np.nan
                    if var2:
                        mp_events[var2][j] = np.nan
        
        return mp_events

    def process_trials(self):
        """Processes all trials and separates by task type"""
        self.data = self.filtered_full_words.copy()
        self.timestamps = self.filtered_timestamps.copy()
        
        # Get unique blocks
        total_blocks = np.unique(self.filtered_ttl_blocks)
        self.block_type_array = np.full(self.filtered_ttl_blocks.shape, '', dtype='O')
        
        # Group blocks by task type
        task_blocks = {
            'dots3DMP': [],
            'dots3DMPtuning': [],
            'dots3DMP_fixedRT': [],
            'SaccadeTraining': []
        }
        
        for block in total_blocks:
            if not np.isin(block, self.pldaps_filetimes).any():
                continue
            
            matched_idx = np.where(block == self.pldaps_filetimes)[0]
            corresponding_par = self.par[matched_idx[0]].strip()
            
            block_indices = np.where(self.filtered_ttl_blocks == block)[0]
            self.block_type_array[block_indices] = corresponding_par
            
            if corresponding_par in task_blocks:
                task_blocks[corresponding_par].append(block_indices)
        
        # Process each task type
        self.event_data = {}
        
        for task_type, blocks_list in task_blocks.items():
            if len(blocks_list) == 0:
                continue
                
            print(f"Processing {task_type}: {len(blocks_list)} blocks")
            
            all_task_events = []
            for block_indices in blocks_list:
                if task_type == 'SaccadeTraining':
                    task_events = self.process_saccade_trials(block_indices)
                else:
                    task_events = self.process_dots3DMP_trials(block_indices, task_type)
                
                if task_events is not None:
                    all_task_events.append(task_events)
            
            # Merge all blocks of the same task type (INDENTED INSIDE THE LOOP!)
            if all_task_events:
                merged_events = {}
                for key in all_task_events[0].keys():
                    merged_events[key] = np.concatenate([evt[key] for evt in all_task_events])
                
                self.event_data[task_type] = merged_events
                print(f"{task_type}: {len(merged_events['goodtrial'])} trials, "
                    f"{np.sum(merged_events['goodtrial'])} good trials")
            else:
                print(f"⚠ {task_type}: No valid trials found across {len(blocks_list)} blocks")

    def save_events_to_mat(self):
            """Saves event data to MATLAB .mat file, organized by task type"""
            # Use same naming convention as CreateUnitStruct
            if self.probe_name in ['ProbeA']:
                save_path = os.path.join(self.data_path, f"{self.subject}{self.date}dots3DMP.mat")
            else:
                save_path = os.path.join(self.data_path, f"{self.subject}{self.date}dots3DMP_{self.probe_name}.mat")
            
            if os.path.exists(save_path):
                mat_contents = scipy.io.loadmat(save_path, simplify_cells=True)
                data_dict = mat_contents['data']
                print(f"Loading existing file with keys: {list(data_dict.keys())}")
            else:
                data_dict = {}
                print("Generating a new file.")
            
            for task_type, events in self.event_data.items():
                if task_type not in data_dict or not isinstance(data_dict[task_type], dict):
                    data_dict[task_type] = {}
                
                data_dict[task_type]['events'] = events
                print(f"  {task_type} now contains: {list(data_dict[task_type].keys())}")
            
            scipy.io.savemat(save_path, {'data': data_dict})
            print(f"Event data saved at '{save_path}'")

class CreateUnitStruct:
    def __init__(self, directory, subject, date, kilosort, AP_name='ProbeA'):
        self.directory = directory
        self.subject = subject
        self.date = date
        self.kilosort = kilosort
        self.AP_name = AP_name
        self.data_path = os.path.join(directory, date)
        self.kilosort_path = os.path.join(directory, date, kilosort)
        
        self.spike_time = np.load(os.path.join(self.kilosort_path, f"spike_times.npy"))
        self.spike_clusters = np.load(os.path.join(self.kilosort_path, f"spike_clusters.npy"))
        # self.channel_positions = np.load(os.path.join(self.kilosort_path, f"channel_positions.npy"))
        self.cluster_group = pd.read_csv(os.path.join(self.kilosort_path, "cluster_info.tsv"), sep='\t')

        # Apply naming convention based on AP_name
        if AP_name in ['ProbeA']:
            self.timestamps = np.load(os.path.join(self.data_path, f"{subject}{date}dots3DMP_APtimestamps.npy"))
            self.AP_blocks = np.load(os.path.join(self.data_path, f"{subject}{date}dots3DMP_APblocks.npy"))
        else:
            self.timestamps = np.load(os.path.join(self.data_path, f"{subject}{date}dots3DMP_AP{AP_name}timestamps.npy"))
            self.AP_blocks = np.load(os.path.join(self.data_path, f"{subject}{date}dots3DMP_AP{AP_name}blocks.npy"))

        self.pldaps_filetimes, self.par, self.par_type = self.check_trial_par()

    def check_trial_par(self):
        """Get the trial type"""
        print("Extracting unit from kilosort...")
        mat_file = os.path.join(self.data_path, f"{self.subject}{self.date}dots3DMP_info.mat")

        mat_contents = scipy.io.loadmat(mat_file, simplify_cells=True)
        info = mat_contents['info']
        par_raw = info['par']
        par = np.array([p.strip() if isinstance(p, str) else p for p in par_raw])
        par_type = np.unique(par)

        return info['pldaps_filetimes'], par, par_type 

    def build_unit_structure(self):
        uniq_spike_clusters = np.unique(self.spike_clusters)

        unit_struct = {
            par_type: {
                'depth': [], 'cluster_id': [], 'groups': [], 'spiketimes': []
            } for par_type in self.par_type
        }

        for par_type in self.par_type:

            trial_indices = [i for i, p in enumerate(self.par) if p == par_type]

            if len(trial_indices) == 0:
                print("No trial type", par_type)
                continue

            filetime_set = set(self.pldaps_filetimes[i] for i in trial_indices)
            blk_idx = np.isin(self.AP_blocks, list(filetime_set))

            spiketimes = []
            cluster_id = []
            depth = []
            group = []

            for i, cluster in enumerate(uniq_spike_clusters):
                idx = np.where((self.spike_clusters == cluster))[0]
            
                if len(idx) == 0:
                    continue
                unit_kilo_frames = self.spike_time[idx]
                keep_mask = blk_idx[unit_kilo_frames]
                unit_kilo_frames = unit_kilo_frames[keep_mask]

                unit_spike_time = self.timestamps[unit_kilo_frames]

                cg_row = self.cluster_group.loc[self.cluster_group["cluster_id"] == cluster]
                if cg_row.empty:
                    continue

                cluster_id.append(cluster)
                depth.append(cg_row["depth"].values[0])
                group.append(int(cg_row["group"].values[0] == "good"))
                spiketimes.append(unit_spike_time)
                
            spiketimes_cells = np.array(spiketimes, dtype=object).reshape(-1, 1)
            spiketimes_cells_transposed = spiketimes_cells.T 
            unit_struct[par_type] = {
                'depth': np.array(depth),
                'cluster_id': np.array(cluster_id),
                'groups': np.array(group),
                'spiketimes': np.array(spiketimes_cells_transposed, dtype=object),
                'cluster_group': self.cluster_group.to_dict(orient='list')
            }

        self.unit_struct = unit_struct

    def save_units_to_mat(self):
        """ Saves unit data to a MATLAB `.mat` file """
        if self.AP_name in ['ProbeA']:
            save_path = os.path.join(self.data_path, f"{self.subject}{self.date}dots3DMP.mat")
        else:
            save_path = os.path.join(self.data_path, f"{self.subject}{self.date}dots3DMP_{self.AP_name}.mat")

        # Robust loading of existing file
        data_dict = {}
        if os.path.exists(save_path):
            try:
                mat_contents = scipy.io.loadmat(save_path, simplify_cells=True)
                if 'data' in mat_contents:
                    data_dict = mat_contents['data']
                    # Ensure it's a dict (sometimes MATLAB saves weird structures)
                    if not isinstance(data_dict, dict):
                        data_dict = {}
                    print(f"✓ Loaded existing file with keys: {list(data_dict.keys())}")
                else:
                    print("⚠ File exists but no 'data' field found, starting fresh")
            except Exception as e:
                print(f"⚠ Error loading file: {e}, starting fresh")
        else:
            print("Creating new file")

        # Update with unit data (preserves existing keys like 'events', 'eyelink')
        for p_type in self.par_type:
            if p_type not in data_dict:
                data_dict[p_type] = {}
            elif not isinstance(data_dict[p_type], dict):
                data_dict[p_type] = {}
            
            data_dict[p_type]['unit'] = self.unit_struct[p_type]
            print(f"  {p_type} now contains: {list(data_dict[p_type].keys())}")

        # Save with compression (smaller files)
        scipy.io.savemat(save_path, {'data': data_dict}, do_compression=True)
        print(f"✓ Unit data saved at '{save_path}'")

class CreateEyeXYStruct:
    def __init__(self, directory, subject, date):
        self.directory = directory
        self.subject = subject
        self.date = date
        self.data_path = os.path.join(directory, date)

        self.eyeXY = np.fromfile(os.path.join(self.data_path, f"{subject}{date}dots3DMP_eyeXY.dat"), dtype='int16')
        self.eyeXY = np.reshape(self.eyeXY, (self.eyeXY.size // 4, 4))

        self.eyeXY_timestamps = np.load(os.path.join(self.data_path, f"{subject}{date}dots3DMP_eyeXYtimestamps.npy"))
        self.eyeXY_blocks = np.load(os.path.join(self.data_path, f"{subject}{date}dots3DMP_eyeXYblocks.npy"))

        self.pldaps_filetimes, self.par, self.par_type = self.check_trial_par()

    def check_trial_par(self):
        """Get the trial type"""
        mat_file = os.path.join(self.data_path, f"{self.subject}{self.date}dots3DMP_info.mat")

        mat_contents = scipy.io.loadmat(mat_file, simplify_cells=True)
        info = mat_contents['info']
        par_raw = info['par']
        par = np.array([p.strip() if isinstance(p, str) else p for p in par_raw])
        par_type = np.unique(par)

        return info['pldaps_filetimes'], par, par_type 

    def build_eyeXY_structure(self):

        eyeXY_struct = {
            par_type: {
                'eyeXY': [], 'timestamps': []
            } for par_type in self.par_type
        }


        for par_type in self.par_type:

            trial_indices = [i for i, p in enumerate(self.par) if p == par_type]

            if len(trial_indices) == 0:
                print("No trial type", par_type)
                continue

            filetime_set = set(self.pldaps_filetimes[i] for i in trial_indices)
            blk_idx = np.isin(self.eyeXY_blocks, list(filetime_set))

            if np.sum(blk_idx) == 0:
                print(f"No matching eyeXY data for trial type {par_type}")
                continue

            eyeXY_data = self.eyeXY[blk_idx, :2]           
            eyeXY_timestamps = self.eyeXY_timestamps[blk_idx]

            eyeXY_struct[par_type]['eyeXY'] = eyeXY_data
            eyeXY_struct[par_type]['timestamps'] = eyeXY_timestamps


        self.eyeXY_struct = eyeXY_struct

        

    def save_eyeXY_to_mat(self):
        """ Saves eyeXY data to a MATLAB `.mat` file and reports trial counts. """
        # Use same naming convention as CreateUnitStruct
        if self.probe_name in ['ProbeA']:
            save_path = os.path.join(self.data_path, f"{self.subject}{self.date}dots3DMP.mat")
        else:
            save_path = os.path.join(self.data_path, f"{self.subject}{self.date}dots3DMP_{self.probe_name}.mat")

        if os.path.exists(save_path):
            mat_contents = scipy.io.loadmat(save_path, simplify_cells=True)
            data_dict = mat_contents['data']
            print(f"Loading existing file with keys: {list(data_dict.keys())}")
        else:
            data_dict = {}
            print("Generating a new file.")

        for p_type in self.par_type:
            if p_type not in data_dict or not isinstance(data_dict[p_type], dict):
                data_dict[p_type] = {}

            data_dict[p_type]['eyelink'] = self.eyeXY_struct[p_type]
            print(f"  {p_type} now contains: {list(data_dict[p_type].keys())}")

        scipy.io.savemat(save_path, {'data': data_dict})
        print(f"EyeXY data saved at '{save_path}'")
    



if __name__ == "__main__":
    directory = "D:\\"
    subject = "zarya"
    date = "20250306"
    kilo = "kilosort4_phy"

    # Run the MergeRecordingFile class
    processor = MergeRecordingFile(directory, subject, date)
    processor.check_electrode_consistency()
    processor.merge_ap_data()
    processor.merge_ttl_data()

    # Run the new CreateEventStruct class
    event_processor = CreateEventStruct(directory, subject, date)
    event_processor.filter_events()
    event_processor.load_info_data()
    event_processor.process_trials()
    event_processor.save_to_mat()

    # Run the new CreateUnitStruct class
    unit_processor = CreateUnitStruct(directory, subject, date, kilo)
    unit_processor.build_unit_structure()
    unit_processor.save_units_to_mat()

