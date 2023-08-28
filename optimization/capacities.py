import datetime
from datetime import date
from enum import Enum
from typing import List, Dict, Union, Callable, Optional

import pandas as pd

# WARNING: Real data has been removed for data protection reasons. Definition below just captures the shape of the data.
raw_caps = {'2012': {'affiliate 1': 100, 'affiliate 2': 50, '...': 80},
            '2013': {'affiliate 1': 100, 'affiliate 2': 50, '...': 80},
            '2014': {'affiliate 1': 100, 'affiliate 2': 50, '...': 80},
            '2015': {'affiliate 1': 100, 'affiliate 2': 50, '...': 80},
            '2015 revised': {'affiliate 1': 100, 'affiliate 2': 50, '...': 80},
            '2016': {'affiliate 1': 100, 'affiliate 2': 50, '...': 80},
            '2016 revised': {'affiliate 1': 100, 'affiliate 2': 50, '...': 80},
            '2017': {'affiliate 1': 100, 'affiliate 2': 50, '...': 80},
            '2017 revised': {'affiliate 1': 100, 'affiliate 2': 50, '...': 80},
            '2018': {'affiliate 1': 100, 'affiliate 2': 50, '...': 80},
            '2018 revised': {'affiliate 1': 100, 'affiliate 2': 50, '...': 80},
            '2019': {'affiliate 1': 100, 'affiliate 2': 50, '...': 80},
            '2020': {'affiliate 1': 100, 'affiliate 2': 50, '...': 80},
            '2020 revised': {'affiliate 1': 100, 'affiliate 2': 50, '...': 80}}

revision_dates = {2017: date(2017, 3, 2), 2018: date(2018, 1, 11), 2020: date(2020, 4, 29)}

def get_standardized_caps(query_date: Union[date, pd.Timestamp]):
    if query_date < date(2011, 10, 1):
        raise ValueError("Too early (I don't have caps for 2011).")
    for fiscal_year in range(2012, 2021):
        if query_date < date(fiscal_year, 10, 1):
            if fiscal_year in revision_dates and query_date >= revision_dates[fiscal_year]:
                return raw_caps[f"{fiscal_year} revised"]
            else:
                return raw_caps[str(fiscal_year)]
    raise ValueError("Too recent.")


class CapacitiesDynamic(Enum):
    changing = 0
    static_end_of_FY = 1
    static_start_of_FY = -1


class Capacities:
    def __init__(self, capacities_function: Callable[[datetime.date], Dict[str, int]]):
        self.get_capacities_at_date = capacities_function


class ConstantCapacities(Capacities):
    def __init__(self, capacities: Dict[str, int]):
        self.capacities = capacities
        super().__init__(lambda _: capacities)

    def get_capacities(self):
        return self.capacities


def _get_caps(query_date: Union[date, pd.Timestamp], localities: List[str], hundredten_percent: bool) -> Dict[str, int]:
    standardized_caps = get_standardized_caps(query_date)
    caps = {}
    if hundredten_percent:
        multiplier = 1.1
    else:
        multiplier = 1.
    for loc in localities:
        if loc in standardized_caps:
            caps[loc] = round(multiplier * standardized_caps[loc])
        else:
            caps[loc] = 0

    return caps


def _apply_caps_minimums(capacities: Dict[str, int], minimums: Optional[Dict[str, int]]):
    if minimums is None:
        minimums = {}

    for loc, amount in minimums.items():
        capacities[loc] = max(capacities[loc], amount)


def get_fy_caps(fiscal_year: int, localities: List[str], hundredten_percent: bool, dynamic_mode: CapacitiesDynamic,
                end_of_year_minimums: Optional[Dict[str, int]] = None) -> Capacities:
    original_capacities = _get_caps(date(fiscal_year - 1, 10, 1), localities, hundredten_percent)
    final_capacities = _get_caps(date(fiscal_year, 9, 30), localities, hundredten_percent)
    _apply_caps_minimums(final_capacities, end_of_year_minimums)
    if dynamic_mode is CapacitiesDynamic.static_end_of_FY or \
            (dynamic_mode is CapacitiesDynamic.changing and fiscal_year not in revision_dates):
        return ConstantCapacities(final_capacities)
    elif dynamic_mode is CapacitiesDynamic.static_start_of_FY:
        assert end_of_year_minimums is None
        return ConstantCapacities(original_capacities)
    else:

        def _getter(query_date):
            if query_date < revision_dates[fiscal_year]:
                return original_capacities
            else:
                return final_capacities

        return Capacities(_getter)
