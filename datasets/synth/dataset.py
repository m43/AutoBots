import os

import numpy as np
from torch.utils.data import Dataset


class SynthV1Dataset(Dataset):
    def __init__(self, dset_path, filename):
        # TODO: Note that the number of agents is hardcoded to match the
        #       standard Synth-v1. This parameter as well as the dataset
        #       preprocessing parameter `--max-number-of-agents` must be
        #       tweaked if a different dataset is used, e.g., one of the
        #       out-of-distribution datasets related to Synth-v1. 
        self.num_others = 11

        self.pred_horizon = 12
        self.num_agent_types = 1  # code assuming only one type of agent (pedestrians).
        self.in_seq_len = 8
        self.predict_yaw = False
        self.map_attr = 0  # dummy
        self.k_attr = 2
        dataset_path = os.path.join(dset_path, filename)
        print(f"SynthV1Dataset: Loading dataset from {os.path.abspath(dataset_path)}")
        self.agents_dataset = np.load(dataset_path)[:, :, :self.num_others + 1]

    def unpack_datapoint(self, trajectories):
        assert len(trajectories) == self.in_seq_len + self.pred_horizon

        # Remove nan values and add mask column to state
        data_mask = np.ones((trajectories.shape[0], trajectories.shape[1], 3))
        data_mask[:, :, :2] = trajectories
        nan_indices = np.where(np.isnan(trajectories[:, :, 0]))
        data_mask[nan_indices] = [0, 0, 0]

        # Separate past and future.
        agents_in = data_mask[:self.in_seq_len]
        agents_out = data_mask[self.in_seq_len:]

        ego_in = agents_in[:, 0]
        ego_out = agents_out[:, 0]

        agent_types = np.ones((self.num_others + 1, self.num_agent_types))
        roads = np.ones((1, 1))  # for dataloading to work with other datasets that have images.

        return ego_in, ego_out, agents_in[:, 1:], agents_out[:, 1:], roads, agent_types

    def __getitem__(self, idx: int):
        trajectories = self.agents_dataset[idx]
        return self.unpack_datapoint(trajectories)

    def __len__(self):
        return len(self.agents_dataset)
