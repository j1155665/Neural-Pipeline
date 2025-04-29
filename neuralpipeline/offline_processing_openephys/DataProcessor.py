import os
import re
import numpy as np
import scipy.io
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

    def find_matching_files(self):
        """Finds recording session folders based on date pattern."""
        files_path = os.path.join(self.directory, f"{self.date}")
        files = os.listdir(files_path)
        matched_files = [f for f in files if self.pattern.match(f)]
        if not matched_files:
            raise ValueError(f"No files found for subject {self.subject} on {self.date}.")
        return matched_files

    def find_matching_data(self):
        """Finds the paths to recording data inside matched session folders."""
        matching_data = []
        for file in self.matching_files:
            data_directory = os.path.join(self.directory, self.date , file, "Record Node 101")
            print(data_directory)
            if not os.path.exists(data_directory):
                continue
            for root, dirs, files in os.walk(data_directory):
                path_parts = root.split(os.sep)

                if len(path_parts) >= 2 and "recording" in path_parts[-1].lower():
                    matching_data.append(root)
        matching_data.sort()
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
    
    def extract_channel_positions(self):
        """Get the configuration of the probe as ProbeConfig.txt"""

        xml_file = os.path.join(self.directory, self.date , self.matching_files[0], 'Record Node 101', 'settings.xml')
        tree = ET.parse(xml_file)
        root = tree.getroot()

        probe_node = root.find(".//NP_PROBE")

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

        file_path = os.path.join(self.directory, self.date ,"ProbeConfig.txt")
        with open(file_path, 'w') as f:
            f.write("X-pos\n")
            f.write(f"[{', '.join(str(x) for x in x_pos)}]\n\n")
            f.write("Y-pos\n")
            f.write(f"[{', '.join(str(y) for y in y_pos)}]\n\n")

        print(f"Positions saved to '{file_path}'.")

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

    def merge_ap_data(self, num_channels=385):
        """Merges AP (action potential) data from multiple recording files and includes block information."""
        output_path = os.path.join(self.directory, f"{self.date}", f"{self.subject}{self.date}dots3DMP_AP.dat")
        aptime_out_path = os.path.join(self.directory, f"{self.date}", f"{self.subject}{self.date}dots3DMP_APtimestamps.npy")
        aptime_block_path = os.path.join(self.directory, f"{self.date}", f"{self.subject}{self.date}dots3DMP_APblocks.npy")

        # Lists to store AP timestamps and block markers
        ap_timestamps_list = []
        ap_block_list = []
        
        final_original_size = 0

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        for i, file in enumerate(self.matching_data):
            print(f"Processing file {i+1}/{len(self.matching_data)}: {file}")
            
            # Path to continuous data
            filepath = os.path.join(file, 'continuous', 'Neuropix-PXI-100.ProbeA-AP', 'continuous.dat')

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
                start_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
                data.tofile(f)  # Write or append data
                end_size = os.path.getsize(output_path)
                print(f"{'Written' if i == 0 else 'Appended'} {file}, Size: {end_size / (1024 ** 3):.2f} GB")

            # Load AP timestamps and add block info
            apt_filepath = os.path.join(file, 'continuous', 'Neuropix-PXI-100.ProbeA-AP', 'timestamps.npy')
            if os.path.exists(apt_filepath):
                ap_timestamps = np.load(apt_filepath)
                ap_timestamps_list.append(ap_timestamps)
                ap_block = np.ones(ap_timestamps.shape) * (i + 1)  # Add block number
                ap_block_list.append(ap_block)
            else:
                print(f"Warning: AP timestamps file not found at {file}")
        
        # Merge AP timestamps and save
        if ap_timestamps_list:
            merged_ap_timestamps = np.concatenate(ap_timestamps_list)
            np.save(aptime_out_path, merged_ap_timestamps)
            np.save(aptime_block_path, np.concatenate(ap_block_list))
            print(f"Merged AP timestamps saved to {aptime_out_path}")
        else:
            print("No AP timestamps to merge.")  
        
        # Final file size and size verification
        final_size = os.path.getsize(output_path) / (1024 ** 3)  # Size in GB
        print(f"Final merged file size: {final_size:.2f} GB")
        
        if abs(final_size - final_original_size) < 1e-6:  
            print("Final size matches the total original size.")
        else:
            print(f"WARNING: Size mismatch! Difference: {final_size - final_original_size:.2f} GB")

        return output_path, aptime_out_path, aptime_block_path


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
            print(f"Processing file {i+1}/{len(self.matching_data)}: {file}")

            ttl_filepath = os.path.join(file, 'events', 'NI-DAQmx-104.PXIe-6341', 'TTL', 'full_words.npy')
            ttlt_filepath = os.path.join(file, 'events', 'NI-DAQmx-104.PXIe-6341', 'TTL', 'timestamps.npy')

            # Load TTL data and add block markers
            if os.path.exists(ttl_filepath):
                ttl_data = np.load(ttl_filepath)
                ttl_list.append(ttl_data)
                ttl_block = np.ones(ttl_data.shape) * (i + 1)  # Add block number for each file
                ttl_block_list.append(ttl_block)
            else:
                print(f"Warning: TTL file not found at {file}")

            # Load TTL timestamps
            if os.path.exists(ttlt_filepath):
                ttl_timestamps = np.load(ttlt_filepath)
                ttl_timestamps_list.append(ttl_timestamps)
            else:
                print(f"Warning: TTL timestamps file not found at {file}")

        # Merge TTL data and save
        if ttl_list:
            merged_ttl = np.concatenate(ttl_list)
            np.save(ttl_out_path, merged_ttl)
            np.save(ttl_block_path, np.concatenate(ttl_block_list))
            print(f"Merged TTL saved to {ttl_out_path}")
        else:
            print("No TTL data to merge.")

        # Merge TTL timestamps and save
        if ttl_timestamps_list:
            merged_ttl_timestamps = np.concatenate(ttl_timestamps_list)
            np.save(ttltime_out_path, merged_ttl_timestamps)
            print(f"Merged TTL timestamps saved to {ttltime_out_path}")
        else:
            print("No TTL timestamps to merge.")

        return ttl_out_path, ttl_block_path, ttltime_out_path


class CreateEventStruct:
    def __init__(self, directory, subject, date):
        self.directory = directory
        self.subject = subject
        self.date = date
        self.data_path = os.path.join(directory, date)
        
        self.full_words = np.load(os.path.join(self.data_path, f"{subject}{date}dots3DMP_ttl.npy"))
        self.timestamps = np.load(os.path.join(self.data_path, f"{subject}{date}dots3DMP_ttltimestamps.npy"))
        self.ttl_blocks = np.load(os.path.join(self.data_path, f"{subject}{date}dots3DMP_ttlblocks.npy"))
        
        self.full_words[self.full_words >= 256] -= 256
        self.block_indices = self.ttl_blocks >= 0  # Include all blocks
        self.block_type = ['dots3DMP', 'dots3DMPtuning']
        self.event_data = None

        # Define event codes as instance variables
        self.TRIAL = 1
        self.FIX = 2
        self.FIXATION = 3
        self.STIMONOFF = 5
        self.SACC = 6
        self.TARGHOLD = 7
        self.POSTTARGHOLD = 8
        self.BREAKFIX = 10

    def filter_events(self):
        """ Filters event timestamps and words based on valid indices. """
        valid_indices = (self.full_words > 0) & (self.full_words != 13)

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

    def process_trials(self):
        """Processes trials, extracts event timestamps, and assigns behavioral indices."""
        self.data = self.filtered_full_words.copy()
        self.timestamps = self.filtered_timestamps.copy()

        # Define event codes
        TRIAL, FIX, FIXATION, STIMONOFF, SACC, TARGHOLD, POSTTARGHOLD, BREAKFIX = 1, 2, 3, 5, 6, 7, 8, 10

        # Define BLOCK and event indices
        total_blocks = np.unique(self.filtered_ttl_blocks)
        self.block_type = np.full(self.filtered_ttl_blocks.shape, '', dtype='O')

        for i in total_blocks:
            block_indices = np.where(self.filtered_ttl_blocks == i)[0]
            data_block = self.data[block_indices[0]:block_indices[-1] + 1]
            first_fix_idx = np.where(data_block == FIX)[0]

            if first_fix_idx.size == 0:
                continue

            PDS_filetime = data_block[1] * 100 + (data_block[first_fix_idx[0] - 1])
            matched_idx = np.where(self.pldaps_filetimes == PDS_filetime)[0]

            if matched_idx.size > 0:
                corresponding_par = self.par[matched_idx[0]]
                self.block_type[block_indices[0]:block_indices[-1] + 1] = corresponding_par
            else:
                print(f"No match found for PDS_filetime: {PDS_filetime}")
                break

        # Define TRIAL
        trial_indices = np.where(self.data == TRIAL)[0]
        idx_diffs = np.diff(trial_indices) > 10
        self.trial_indices = np.concatenate(([trial_indices[0]], trial_indices[1:][idx_diffs]))
        self.block_type = np.array([b.strip() for b in self.block_type[self.trial_indices]])
        self.trial_size = len(self.trial_indices)

        # Define Behavior Indices
        event_indices = {
            'fpOn_idx': self.timestamps[np.where(self.data == FIX)[0]],
            'fixation_idx': self.timestamps[np.where(self.data == FIXATION)[0]],
            'stimOn_idx': self.timestamps[np.where(self.data == STIMONOFF)[0]],
            'stimOff_idx': self.timestamps[np.where(self.data == STIMONOFF)[0]],
            'saccOnset_idx': self.timestamps[np.where(self.data == SACC)[0]],
            'targHold_idx': self.timestamps[np.where(self.data == TARGHOLD)[0]],
            'postTargHold_idx': self.timestamps[np.where(self.data == POSTTARGHOLD)[0]],
            'breakFix_idx': self.timestamps[np.where(self.data == BREAKFIX)[0]]
        }

        # Initialize event_data dictionary
        self.event_data = {key: np.full(self.trial_size, np.nan) for key in [
            'fpOn', 'fixation', 'stimOn', 'stimOff',
            'saccOnset', 'targHold', 'postTargHold', 'breakFix',
            'good_trial', 'headingInd', 'modality', 'coherenceInd',
            'deltaInd', 'choice', 'correct', 'PDW'
        ]}

        vars_to_process = [
            ('deltaInd', 'PDW'),
            ('coherenceInd', 'correct'),
            ('modality', 'choice'),
            ('headingInd',)
        ]

        # Process each trial
        for j in range(self.trial_size):
            current_trial = self.timestamps[self.trial_indices[j]]
            previous_trial = self.timestamps[0] if j == 0 else self.timestamps[self.trial_indices[j - 1]]

            for key, idx in event_indices.items():
                valid_idx = idx[(idx > previous_trial) & (idx < current_trial)]
                if key == 'stimOff_idx':
                     self.event_data[key.replace("_idx", "")][j] = valid_idx[-1] if valid_idx.size > 0 else np.nan
                else:
                        self.event_data[key.replace("_idx", "")][j] = valid_idx[0] if valid_idx.size > 0 else np.nan

            if np.isnan(self.event_data['breakFix'][j]):
                self.event_data['good_trial'][j] = 1
            else:
                self.event_data['good_trial'][j] = 0

            if np.isnan(self.event_data['fixation'][j]):
                self.event_data['good_trial'][j] = 0

            if j == self.trial_size - 1:
                event_info = self.data[self.trial_indices[j]:]
            else:
                event_info = self.data[self.trial_indices[j]:self.trial_indices[j + 1]]

            event_info -= 70

            for idx, (var1, *var2) in enumerate(vars_to_process):
                if idx > 0:
                    event_info -= 10
                if "headingInd" in var1 or any("headingInd" in v for v in var2):
                    new_idx = (event_info <= 20) & (event_info >= 0)
                else:
                    new_idx = (event_info <= 10) & (event_info >= 0)

                new_info = event_info[new_idx]

                if new_info.size > 0:
                    self.event_data[var1][j] = new_info[0]
                    if var2:
                        self.event_data[var2[0]][j] = new_info[-1]
                        if self.event_data['breakFix'][j] == 0:
                            self.event_data[var2[0]][j] = np.nan
                else:
                    self.event_data[var1][j] = np.nan
                    if var2:
                        self.event_data[var2[0]][j] = np.nan

    def save_to_mat(self):
        """ Saves event data to a MATLAB `.mat` file and reports trial counts. """
        save_path = os.path.join(self.data_path, f"{self.subject}{self.date}dots3DMP.mat")
        par_type = ['dots3DMP', 'dots3DMPtuning']
        data = {}

        for p_type in par_type:
            trial_2_save = self.block_type == p_type
            trial_data = {}
            
            for key in self.event_data:
                if p_type == 'dots3DMPtuning' and key in ['choice', 'correct', 'PDW']:
                    continue
                trial_data[key] = self.event_data[key][trial_2_save]

            total_trials = np.size(self.event_data['good_trial'][trial_2_save])
            good_trials = int(np.sum(self.event_data['good_trial'][trial_2_save]))

            data[p_type] = {'events': trial_data}
            
            print(f"{p_type}: Total Trials = {total_trials}, Good Trials = {good_trials}")

        scipy.io.savemat(save_path, {'data': data})
        print(f"Data saved at '{save_path}'")


if __name__ == "__main__":
    directory = "D:\\"
    subject = "zarya"
    date = "20250306"

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
