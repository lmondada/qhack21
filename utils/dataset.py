import pandas as pd
from dataclasses import dataclass
from collections import defaultdict
from copy import copy
from random import shuffle

from .data_handling import load_data, get_combined_cov_pos
from .utils import print_progress

@dataclass
class Dataset:
    data: pd.DataFrame

    def get_cov_triplets(self):
        self.triplets = []
        for event_id, cdms in self.filtered_data.groupby("event_id"):
            sorted_cdms = cdms.sort_values(by="time_to_tca")
            covs = pd.DataFrame({
                "cov": get_combined_cov_pos(sorted_cdms),
                "time_to_tca": sorted_cdms.time_to_tca
            })
            oneday = lambda x, y: 0.9 < y.time_to_tca - x.time_to_tca < 1.1
            for _, cov1 in covs.iterrows():
                for _, cov2 in covs.iterrows():
                    if not oneday(cov1, cov2):
                        continue
                    for _, cov3 in covs.iterrows():
                        if not oneday(cov2, cov3):
                            continue
                        self.triplets.append((cov1['cov'], cov2['cov'], cov3['cov']))

    def filter_data(self):
        self.filtered_data = self.data[self.data.mission_id == 5]

    def process_data(self):
        with print_progress("Filtering dataset"):
            self.filter_data()
        with print_progress("Computing covmats"):
            self.get_cov_triplets()

    def get_train_and_test(self):
        self.process_data()
        shuffle(self.triplets)
        
        splitat = len(self.triplets) * 8 // 10
        train_data = self.triplets[:splitat]
        test_data  = self.triplets[splitat:]

        print(f"Returning dataset with {len(train_data)}/{len(test_data)} training/testing datapoints")
        return train_data, test_data
