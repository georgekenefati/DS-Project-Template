from glob import glob
import pickle
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import mne
import os
from mne.time_frequency import tfr_array_morlet, AverageTFRArray
import seaborn as sns
from typing import Dict, List, Union
from IPython.display import clear_output

sns.set(style="white", font_scale=1.5)


class Subject:
    def __init__(self, subject_id: str):
        assert isinstance(subject_id, str), "Subject ID must be a string"
        self.subject_id = subject_id
        self.response = None

        sub_ids = {
            "CP": [
                "018",
                "022",
                "024",
                "031",
                "032",
                "034",
                "036",
                "039",
                "040",
                "045",
                "046",
                "052",
                "020",
                "021",
                "023",
                "029",
                "037",
                "041",
                "042",
                "044",
                "048",
                "049",
                "050",
                "056",
            ],
            "HC": [
                "C10",
                "C11",
                "C12",
                "C13",
                "C14",
                "C15",
                "C16",
                "C17",
                "C18",
                "C19",
                "C2.",
                "C24",
                "C25",
                "C26",
                "C27",
                "C3.",
                "C6.",
                "C7.",
                "C9.",
            ],
            "WSP": [
                "018",
                "022",
                "024",
                "031",
                "032",
                "034",
                "036",
                "039",
                "040",
                "045",
                "046",
                "052",
            ],
            "LP": [
                "020",
                "021",
                "023",
                "029",
                "044",
                "037",
                "041",
                "042",
                "048",
                "049",
                "050",
                "056",
            ],
        }

        # Assign group from sub_ids dict keys if sub_id is in any of the keys
        self.group = next((key for key in sub_ids if subject_id in sub_ids[key]), None)

    def __str__(self):
        return f"Subject ID: {self.subject_id}, Group: {self.group}, Response: {self.response}"


class SubjectGroup:
    def __init__(self, subjects: List[Subject]):
        assert isinstance(subjects, list), "Input must be a list"
        assert all(
            [isinstance(el, Subject) for el in subjects]
        ), "Input must be a list of Subjects"
        self.subjects = subjects
        self.group = subjects[0].group

    def __str__(self):
        return f"Subjects: {self.subjects}"


class SubjectProcessor:
    def __init__(self, paths_dict: Dict[str, str], roi_acronyms: List[str]):
        self.yes_list = []
        self.no_list = []
        self.maybe_list = []

        # Define paths and settings
        self.paths_dict = paths_dict
        self.processed_data_path = self.paths_dict["processed_data_path"]
        self.stc_path = self.paths_dict["stc_path"]
        self.EO_resting_data_path = self.paths_dict["EO_resting_data_path"]
        self.zscored_epochs_data_path = self.paths_dict["zscored_epochs_data_path"]

        self.sfreq = 400  # Hz
        self.roi_acronyms = roi_acronyms

    def _fill_nan_channels(self, epochs):
        incomplete_ch_names = epochs.info["ch_names"]
        complete_ch_names = [
            "Fp1",
            "Fpz",
            "Fp2",
            "AF3",
            "AF4",
            "F11",
            "F7",
            "F5",
            "F3",
            "F1",
            "Fz",
            "F2",
            "F4",
            "F6",
            "F8",
            "F12",
            "FT11",
            "FC5",
            "FC3",
            "FC1",
            "FCz",
            "FC2",
            "FC4",
            "FC6",
            "FT12",
            "T7",
            "C5",
            "C3",
            "C1",
            "Cz",
            "C2",
            "C4",
            "C6",
            "T8",
            "TP7",
            "CP5",
            "CP3",
            "CP1",
            "CPz",
            "CP2",
            "CP4",
            "CP6",
            "TP8",
            "M1",
            "M2",
            "P7",
            "P5",
            "P3",
            "P1",
            "Pz",
            "P2",
            "P4",
            "P6",
            "P8",
            "PO7",
            "PO3",
            "POz",
            "PO4",
            "PO8",
            "O1",
            "Oz",
            "O2",
            "Cb1",
            "Cb2",
        ]
        complete_ch_names = [ch_name.upper() for ch_name in complete_ch_names]
        missing_ch_ids = [
            i
            for i in range(len(complete_ch_names))
            if complete_ch_names[i] not in incomplete_ch_names
        ]

        data = epochs.get_data(copy=False)
        data = np.insert(data, missing_ch_ids, np.nan, axis=1)

        info = mne.create_info(
            ch_names=complete_ch_names, sfreq=self.sfreq, ch_types="eeg"
        )

        epochs = mne.EpochsArray(data, info)
        return epochs

    def _load_epochs(self, subject_id: str):
        print(f"\nLoading Epochs for {subject_id}...")
        epo_fname = glob(f"{self.processed_data_path}/{subject_id}*epo.fif")[0]
        epochs = mne.read_epochs(epo_fname)
        assert isinstance(
            epochs, mne.epochs.EpochsFIF
        ), "Input must be an Epochs object"

        if len(epochs.info["ch_names"]) < 64:
            epochs = self._fill_nan_channels(epochs)
        evoked = np.nanmean(epochs.get_data(copy=False), axis=0)
        sem = np.nanstd(epochs.get_data(copy=False), axis=0) / np.sqrt(len(epochs))
        return epochs, evoked, sem

    def _load_stc_epochs(self, subject_id: str):
        print(f"Loading STC epochs for {subject_id}...")
        stc_epo_fname = glob(
            f"{self.zscored_epochs_data_path}/{subject_id}_epochs.pkl"
        )[0]
        stc_epo = pickle.load(open(stc_epo_fname, "rb"))
        stc_epo = np.array(stc_epo)

        stim_fname = glob(f"{self.processed_data_path}/{subject_id}*stim_labels.mat")[0]
        stim_labels = sio.loadmat(stim_fname)["stim_labels"][0]

        print(f"Loaded {len(stim_labels)} stimulus labels")
        print(f"{sum(stim_labels == 3)} hand trials (out of {len(stim_labels)})")

        stc_epo_array = np.nanmean(
            stc_epo[stim_labels == 3], axis=0
        )  # average over hand trials

        assert isinstance(stc_epo_array, np.ndarray), "Input must be an array"
        return stc_epo_array

    def _load_complete_data(
        self,
        subjects: Union[Subject, SubjectGroup],
    ):
        assert isinstance(subjects, Subject) or isinstance(
            subjects, SubjectGroup
        ), "Input must be an instance of Subject or SubjectGroup"

        if isinstance(subjects, SubjectGroup):
            subjects_list = [subject for subject in subjects.subjects]
            print(f"Loading data for {len(subjects_list)} subjects...")
            print(f"Subjects: {[subject.subject_id for subject in subjects_list]}")
        elif isinstance(subjects, Subject):
            subjects_list = [subjects]

        evoked_data_arrays = []
        sem_epochs_per_sub = []
        stc_epo_arrays = []
        stc_resting_arrays = []
        for subject in subjects_list:
            this_sub_id = subject.subject_id
            epochs, evoked, sem = self._load_epochs(this_sub_id)
            evoked_data_arrays.append(evoked)
            sem_epochs_per_sub.append(sem)

            stc_epo_array = self._load_stc_epochs(this_sub_id)
            stc_epo_arrays.append(stc_epo_array)

            stc_resting = None
            stc_resting_arrays.append(stc_resting)

        # combine data across subjects
        stc_epo_array = np.nanmean(np.array(stc_epo_arrays), axis=0)
        if stc_epo_array.ndim != 3:
            stc_epo_array = np.expand_dims(stc_epo_array, axis=0)
        stc_resting = (
            np.nanmean(np.array(stc_resting_arrays), axis=0)
            if stc_resting is not None
            else None
        )
        evoked_data_arrays = np.array(evoked_data_arrays)
        sem_epochs_per_sub = np.array(sem_epochs_per_sub)

        return (
            epochs,
            evoked_data_arrays,
            sem_epochs_per_sub,
            stc_epo_array,
            stc_resting,
        )
