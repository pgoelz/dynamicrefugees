from datetime import datetime
from typing import List, Dict, Optional, Set, Tuple

import pandas as pd
import numpy as np

from annie_ml.dataclean import agents_to_features, historic_data_to_agents
from annie_ml.utils import import_list, models_load, model_features


class Case:
    _check_compatibility = "full"

    _compatibility_info = pd.read_csv("annie_ml/compatibilities.csv", index_col="city").to_dict("index")
    for info in _compatibility_info.values():
        info["nats"] = set(code.lower() for code in info["nats"].split(", "))
        info["slang"] = set(code.lower() for code in info["slang"].split(", "))
        assert info["lfs"] in ("Y", "N")
        info["lfs"] = info["lfs"] == "Y"
        assert info["sps"] in ("Y", "N")
        info["sps"] = info["sps"] == "Y"
    _mismatch_localities = set()

    def __init__(self, size: int, expected_employment: Dict[str, float], ties_to: Optional[str],
                 nationality_code: str, single_parent: bool, language_codes: Set[str]):
        self.size = size
        self.expected_employment = expected_employment
        self.ties_to = ties_to
        self.nationality_code = nationality_code.lower()
        self.single_parent = single_parent
        self.language_codes = set(code.lower() for code in language_codes)

    def __repr__(self):
        return (f"Case({self.size}, {self.expected_employment}, {self.ties_to}, {self.nationality_code}, "
                f"{self.single_parent}, {self.language_codes})")

    def is_compatible(self, locality_name: str):
        if Case._check_compatibility == "none":
            return True
        if self.ties_to is not None:
            return self.ties_to == locality_name
        if Case._check_compatibility == "tied only":
            return True
        if locality_name not in Case._compatibility_info:
            if locality_name not in Case._mismatch_localities:
                print(f"Warning: No compatibility info for {locality_name}, assume compatibility.")
                Case._mismatch_localities.add(locality_name)
            return True
        info = Case._compatibility_info[locality_name]
        if not info["lfs"] and self.size >= 6:
            return False
        if not info["sps"] and self.single_parent:
            return False
        if self.nationality_code not in info["nats"]:
            return False
        if "eng" not in self.language_codes:
            loc_languages = info["slang"]
            if self.language_codes.isdisjoint(loc_languages):
                return False
        return True


class CaseWithArrival(Case):
    def __init__(self, size: int, expected_employment: Dict[str, float], ties_to: Optional[str],
                 nationality_code: str, single_parent: bool, language_codes: Set[str], arrival_date: datetime.date,
                 historic_placement: str):
        self.arrival_date = arrival_date
        self.historic_placement = historic_placement
        super().__init__(size, expected_employment, ties_to, nationality_code, single_parent, language_codes)


def real_year_cases(localities, data_path="~/HIAS_FY06-FY20.csv", min_allocation_date=None, max_allocation_date=None,
                    min_arrival_date=None, max_arrival_date=None, only_free=False) \
        -> Tuple[List[datetime.date], List[List[CaseWithArrival]]]:
    nust_model, ust_model = models_load()
    nust_features, ust_features = model_features()
    synergy_list = import_list()

    df = historic_data_to_agents(data_path, min_allocation_date, max_allocation_date, only_free=only_free)
    if only_free:
        df['ties to'] = ''

    employment_probabilities: Dict[str, Dict[str, float]]
    employment_probabilities = {}
    for locality in localities:
        df_locality = df.copy()
        df_locality["affiliate"] = locality

        features = agents_to_features(df_locality, synergy_list)
        assert len(features) > 0

        nust_X = nust_model['scaler'].transform(features[nust_features])
        nust_Y = nust_model['model'].predict_proba(nust_X)[:, 1]
        ust_X = ust_model['scaler'].transform(features[ust_features])
        ust_Y = ust_model['model'].predict_proba(ust_X)[:, 1]
        employment = np.where(df_locality["ties to"] == "", nust_Y, ust_Y)
        employment[(features['age'] < 18) | (features['age'] > 64)] = 0.
        df_locality["employment"] = employment

        grouped = df_locality.groupby('case number', as_index=True)
        for cn, rows in grouped:
            if cn not in employment_probabilities:
                employment_probabilities[cn] = {}
            employment_probabilities[cn][locality] = rows['employment'].sum()

    completed_cases = set()

    # Potential speedup: instead of pandas grouping: sort -> itertuples -> manually put together
    # For now, I don't do that since this code is not the only performance bottleneck and for readability.
    time_grouped = df.groupby(['allocation date', 'case number'], as_index=True, sort=True)

    batches = []
    dates = []
    for (date, case_number), rows in time_grouped:
        if case_number in completed_cases:
            print(f"Warning: case {case_number} is split across multiple allocation dates. Treat as distinct.")
        completed_cases.add(case_number)

        all_languages = set().union(*rows[rows["age"] >= 18]['languages spoken well'])

        all_nationalities = set(rows['nationality code'])
        assert len(all_nationalities) == 1
        nationality = all_nationalities.pop()

        all_arrivals = set(rows["arrival date"])
        assert len(all_arrivals) == 1
        arrival_date = pd.to_datetime(all_arrivals.pop())
        all_historic = set(rows["historic affiliate"])
        assert len(all_historic) == 1
        historic = all_historic.pop()

        assert rows["seq"].nunique() == len(rows)

        num_parents = np.where(rows["relat code"].isin(["PA", "HU", "WI"]), 1, 0).sum()
        num_children = np.where(~(rows["relat code"].isin(["PA", "HU", "WI"])) & (rows["age"] < 18), 1, 0).sum()
        single_parent = num_parents == 1 and num_children >= 1

        all_ties_to = set(rows['ties to'])
        ties_to = all_ties_to.pop()
        if len(all_ties_to) > 1:
            print(f"Warning: Case {case_number} has ties to multiple places: {all_ties_to}. Ignore all but {ties_to}.")
        if ties_to == '':
            ties_to = None
        elif ties_to not in localities:
            print("Warning: Ignoring case that has ties to unknown locality " + repr(ties_to))
            continue

        case = CaseWithArrival(len(rows), employment_probabilities[case_number], ties_to, nationality, single_parent,
                               all_languages, arrival_date, historic)

        if len(dates) == 0 or date != dates[-1]:
            batches.append([])
            dates.append(date)

        batches[-1].append(case)

    # convert dates from pandas timestamps to datetime.date objects
    dates = [date.date() for date in dates]

    return dates, batches
