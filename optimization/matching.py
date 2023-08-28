import bisect
import datetime
from abc import abstractmethod
from enum import Enum
from math import inf, floor
from random import choices
from typing import List, Optional, Dict

from gurobipy import GRB, Model, quicksum
from scipy.stats import poisson, uniform, nbinom

from optimization.capacities import Capacities
from optimization.cases import Case, CaseWithArrival

EPS = .0001  # tiny bit of extra employment per agent in ILP, to discourage leaving agents unmatched


class KindOfDual(Enum):
    """Strategies of selecting between optimal price vectors."""
    maximal = "max"  # select the element-wise maximal set of prices
    minimal = "min"  # select the element-wise minimal set of prices
    arbitrary = "arbitrary"  # leave the choice of prices up to the LP solver (faster, but unprincipled)


def get_duals(affiliate_names: List[str], cases: List[Case], capacities: Dict[str, int], which_dual: KindOfDual) \
        -> Dict[str, float]:
    """Compute the duals of the following assignment LP:
    maximize Σ_{case, aff compatible} employment_{case, aff} * x_{case, aff}
    s.t.     Σ_{case compatible with aff} size_case * x_{case, aff} <= capacity_aff  ∀ affiliates aff
             Σ_{affiliate aff compatible with case} x_{case, aff} <= 1  ∀ cases case
             x_{case, aff} >= 0  ∀ affiliates aff, cases case compatible with aff

    Args:
        affiliate_names: Names of all affiliates/affiliates.
        cases: All Cases for which to construct the matching LP.
        capacities: Dictionary mapping affiliate names to the number of remaining refugees who can be hosted.
        which_dual: Since there are typically different optimal price vectors, which one should be selected?

    Returns:
        Dictionary mapping affiliate names to their optimal price.
    """
    assert isinstance(which_dual, KindOfDual)

    m = Model()
    m.setParam("OutputFlag", False)  # hide Gurobi debug output

    # We directly solve the dual of the allocation LP. The names are called surplus and price in analogy to the Fisher
    # market, but, since prices are not actually paid, there is no obvious interpretation to, say, the surplus in the
    # refugee setting. The LP looks as follows:
    #
    # minimize Σ_{case in cases} surplus_case + Σ_{aff in affiliate_names} capacity_aff * price_aff
    # s.t.     surplus_case >= employment_{case, aff} - size_case * price_aff  ∀ aff, cases case compatible with aff
    #          surplus_case >= 0, price_aff >= 0  ∀ aff, case

    # declare variables
    surplus_vars = [m.addVar(vtype=GRB.CONTINUOUS, lb=0., ub=inf, name=f"surplus_{i}") for i in range(len(cases))]
    price_vars = {aff: m.addVar(vtype=GRB.CONTINUOUS, lb=0., ub=inf, name=f"price_{aff}") for aff in affiliate_names}

    # define constraints
    for affiliate, price in price_vars.items():
        for case, surplus in zip(cases, surplus_vars):
            if case.is_compatible(affiliate):
                m.addConstr(surplus >= case.expected_employment[affiliate] - case.size * price)
        if capacities[affiliate] == 0:
            m.addConstr(price == 0.)

    # define objective and optimize
    objective = quicksum(surplus_vars) + quicksum(capacities[aff] * price_vars[aff] for aff in affiliate_names)
    m.setObjective(objective, GRB.MINIMIZE)
    m.optimize()
    assert m.status == GRB.OPTIMAL
    optimal_objective = m.objVal

    if which_dual is not KindOfDual.arbitrary:
        # To find, the point-wise minimal (maximal) set of dual prices, add a constraint saying that the objective value
        # of the dual must be the optimal value just computed, and then minimize (maximize) the sum of prices. A priori,
        # it is not obvious that a point-wise minimal (maximal) set of prices exist, but this follows for example from:
        # Faruk Gul and Ennio Stacchetti, “Walrasian Equilibrium with Gross Substitutes,” Journal of Economic Theory 87,
        # no. 1 (1999): 95–124. http://www.princeton.edu/~fgul/walras.pdf

        if which_dual is KindOfDual.minimal:
            sense = GRB.MINIMIZE
        else:
            assert which_dual is KindOfDual.maximal
            sense = GRB.MAXIMIZE
        m.setObjective(quicksum(price_vars.values()), sense)

        # sometimes, numerical imprecision forces us to slightly relax the optimality constraint
        tolerance = 0
        while True:
            optimal_objective_value = m.addConstr(objective <= optimal_objective + tolerance * EPS,
                                                  name="optimal_objective_value")
            m.optimize()

            if m.status in [GRB.INFEASIBLE, GRB.INF_OR_UNBD]:
                tolerance += 1
                print(f"Must relax optimality in getting specific duals, for the {tolerance}th time.")
                m.remove(optimal_objective_value)
            else:
                break

        assert m.status == GRB.OPTIMAL, f"Got weird Gurobi status {m.status}."

    # return the prices resulting from the last optimization to have run
    return {aff: price.x for aff, price in price_vars.items()}


class MatchingAlgorithm:
    deterministic = True

    def __init__(self, locality_names: List[str]):
        self.locality_names = locality_names

    @abstractmethod
    def allocate_batch(self, batch: List[Case], caps: Dict[str, int], date: datetime.date) -> List[Optional[str]]:
        """For each item in the batch, return the index of the affiliate it is allocated to, or None to leave it
        unmatched.
        """
        pass


class OptimumHindsightAlgorithm(MatchingAlgorithm):
    def __init__(self, locality_names, fy_dates, fy_batches, capacities: Dict[str, int]):
        self.fy_dates = fy_dates
        self.fy_batches = fy_batches
        self.t = 0
        super().__init__(locality_names)

        m = Model()
        m.setParam("OutputFlag", False)
        m.setParam("MIPGap", .00001)
        allocation_vars = []
        for batch in fy_batches:
            batch_vars = []
            for case in batch:
                case_vars = {}
                any_compatible = False
                for loc in locality_names:
                    is_compatible = case.is_compatible(loc)
                    any_compatible = any_compatible or is_compatible
                    case_vars[loc] = m.addVar(vtype=GRB.INTEGER, obj=case.expected_employment[loc], lb=0.,
                                              ub=int(is_compatible))  # upper bound prevents incompatible allocation
                if not any_compatible:
                    print(f"Warning: Case {case} is incompatible with all affiliates.")
                m.addConstr(quicksum(case_vars.values()) <= 1)  # assign case to at most one affiliate
                batch_vars.append(case_vars)
            allocation_vars.append(batch_vars)

        for loc, cap in capacities.items():
            size_monomials = []
            for batch, batch_vars in zip(fy_batches, allocation_vars):
                for case, case_vars in zip(batch, batch_vars):
                    size_monomials.append(case.size * case_vars[loc])
            m.addConstr(quicksum(size_monomials) <= cap)  # respect affiliate capacities

        m.modelSense = GRB.MAXIMIZE
        m.optimize()
        assert m.status == GRB.OPTIMAL

        self.unused_capacity = capacities.copy()  # this can be used to greedily match agents with zero employment score
        # where it does not harm employment
        goes_where = []
        for batch, batch_vars in zip(fy_batches, allocation_vars):
            goes_where_batch = []
            for case, case_vars in zip(batch, batch_vars):
                where = [loc for loc in locality_names if case_vars[loc].x > .5]
                assert len(where) <= 1
                if len(where) == 0:
                    goes_where_batch.append(None)
                else:
                    loc = where[0]
                    goes_where_batch.append(loc)
                    self.unused_capacity[loc] -= case.size
            goes_where.append(goes_where_batch)
        self.goes_where = goes_where

    def allocate_batch(self, batch: List[Case], caps: Dict[str, int], date: datetime.date) -> List[Optional[str]]:
        assert self.t < len(self.fy_batches)
        assert date == self.fy_dates[self.t]
        assert batch == self.fy_batches[self.t]

        alloc = []
        for case, where in zip(batch, self.goes_where[self.t]):
            if where is not None:
                alloc.append(where)
            else:
                alt_where = None
                for loc in self.locality_names:
                    if case.is_compatible(loc) and self.unused_capacity[loc] >= case.size:
                        alt_where = loc
                        self.unused_capacity[loc] -= case.size
                        break
                alloc.append(alt_where)

        self.t += 1
        return alloc


class HistoricMatching(MatchingAlgorithm):
    def allocate_batch(self, batch: List[CaseWithArrival], caps: Dict[str, int], date: datetime.date) -> List[
        Optional[str]]:
        return [case.historic_placement for case in batch]


class PriceAlgorithm(MatchingAlgorithm):
    def __init__(self, locality_names, price_multiplier: float = 1.):
        self.price_multiplier = price_multiplier
        super().__init__(locality_names)

    @abstractmethod
    def get_prices(self, batch, caps, date) -> Dict[str, float]:
        pass

    def allocate_batch(self, batch: List[Case], caps: Dict[str, int], date: datetime.date) -> List[Optional[str]]:
        prices = self.get_prices(batch, caps, date)
        assert list(prices) == self.locality_names

        prices = {loc: self.price_multiplier * price for loc, price in prices.items()}

        m = Model()
        m.setParam("OutputFlag", False)
        allocation_vars = []
        for case in batch:
            case_vars = {}
            for locality, price in prices.items():
                is_compatible = case.is_compatible(locality)
                case_vars[locality] = m.addVar(vtype=GRB.INTEGER,
                                               obj=case.expected_employment[locality] + case.size * (EPS - price),
                                               lb=0., ub=int(is_compatible))
            m.addConstr(quicksum(case_vars.values()) <= 1)
            allocation_vars.append(case_vars)

        for loc, cap in caps.items():
            size_monomials = []
            for case, case_vars in zip(batch, allocation_vars):
                size_monomials.append(case.size * case_vars[loc])
            m.addConstr(quicksum(size_monomials) <= cap)

        m.modelSense = GRB.MAXIMIZE
        m.optimize()

        allocation = []
        for case, case_vars in zip(batch, allocation_vars):
            matches = [loc for loc, var in case_vars.items() if var.x > .5]
            if len(matches) == 0:
                allocation.append(None)
            else:
                assert len(matches) == 1
                allocation.append(matches[0])
        return allocation


class GreedyAlgorithm(PriceAlgorithm):
    def get_prices(self, batch, caps, date) -> Dict[str, float]:
        return {loc: 0. for loc in self.locality_names}


class CasePool:
    @abstractmethod
    def get_pool(self, date: datetime.date) -> List[Case]:
        pass


class ConstantCasePool(CasePool):
    def __init__(self, pool: List[Case]):
        self.pool = pool

    def get_pool(self, date: datetime.date) -> List[Case]:
        return self.pool


class LookBackCasePool(CasePool):
    def __init__(self, all_dates, all_batches, delta: datetime.timedelta):
        assert (sorted(all_dates) == all_dates)
        assert (len(all_dates) == len(all_batches))
        self.dates = all_dates
        self.delta = delta
        self.all_cases = [case for batch in all_batches for case in batch]
        self.prefix_case_counts = [0]
        for batch in all_batches:
            self.prefix_case_counts.append(self.prefix_case_counts[-1] + len(batch))
        for i in range(len(all_batches) + 1):
            assert self.prefix_case_counts[i] == sum(len(b) for b in all_batches[:i])
        for i, batch in enumerate(all_batches):
            assert batch == self.all_cases[self.prefix_case_counts[i]:self.prefix_case_counts[i + 1]]

    def get_pool(self, date):
        left = bisect.bisect_left(self.dates, date - self.delta)
        assert self.dates[left] >= date - self.delta
        assert left == 0 or self.dates[left - 1] < date - self.delta
        right = bisect.bisect_right(self.dates, date)
        assert self.dates[right - 1] == date
        assert right == len(self.dates) or self.dates[right] > date
        return self.all_cases[self.prefix_case_counts[left]: self.prefix_case_counts[right]]


class TrajectoryLengths:
    @abstractmethod
    def get_trajectory_lengths(self, current_caps: Dict[str, int], num_trajectories: int, case_count: int,
                               arrival_count: int) -> List[int]:
        pass


class KnownCaseNumberTrajectoryLengths(TrajectoryLengths):
    def __init__(self, num_cases: int):
        self.expected_num_cases = num_cases

    def get_trajectory_lengths(self, current_caps: Dict[str, int], num_trajectories: int, case_count: int,
                               arrival_count: int) -> List[int]:
        assert case_count <= self.expected_num_cases
        return [self.expected_num_cases - case_count for _ in range(num_trajectories)]


class TrustCapacityTrajectoryLengths(TrajectoryLengths):
    def __init__(self, average_case_size: float):
        self.average_case_size = average_case_size

    def get_trajectory_lengths(self, current_caps: Dict[str, int], num_trajectories: int, case_count: int,
                               arrival_count: int) -> List[int]:
        expected_num_refugees = 10 / 11 * sum(current_caps.values())
        expected_remaining_refugees = max(expected_num_refugees - arrival_count, 0)
        expected_remaining_cases = round(expected_remaining_refugees / self.average_case_size)
        return [expected_remaining_cases for _ in range(num_trajectories)]


class PoissonTrustCapacityTrajectoryLengths(TrajectoryLengths):
    def __init__(self, average_case_size: float):
        self.average_case_size = average_case_size

    def get_trajectory_lengths(self, current_caps: Dict[str, int], num_trajectories: int, case_count: int,
                               arrival_count: int) -> List[int]:
        expected_num_refugees = 10 / 11 * sum(current_caps.values())
        expected_num_cases = expected_num_refugees / self.average_case_size

        # probability that Poisson variable with mean total_cases_expected < cases_already_arrived
        # this code will have problems when case_count is substantially larger than expected_num_cases:
        # percentile_cutoff will be very close to 1, up to the point where floating point arithmetic cannot distinguish
        # it from 1 any more; then poisson.ppf will be infinity and this code crashes
        # In principle, it should be possible to avoid this problem by using poisson.sf rather than poisson.cdf,
        # which computes 1-CDF and therefore remains representable as a nonzero floating point number much longer.
        # This indeed works, but isn’t immediately useful because poisson.isf (the equivalent of poisson.ppf taking in
        # 1-percentile rather than percentile) is badly implemented and returns nan for small values.
        # For now, I just avoid computing this experiment with very large arrivals. If ≤ 2500 cases are expected,
        # 110% arrivals barely still seem feasible; for 115%, all methods known to me struggle with imprecision.
        percentile_cutoff = poisson.cdf(case_count - 1, expected_num_cases)
        conditioned_percentiles = 1. - uniform.rvs(scale=1. - percentile_cutoff, size=num_trajectories)
        traj_lengths = []
        for tl in poisson.ppf(conditioned_percentiles, expected_num_cases):
            assert tl >= case_count
            traj_lengths.append(int(tl) - case_count)
        return traj_lengths


class NegBinomialTrustCapacityTrajectoryLengths(TrajectoryLengths):
    def __init__(self, average_case_size: float):
        self.average_case_size = average_case_size

    def get_trajectory_lengths(self, current_caps: Dict[str, int], num_trajectories: int, case_count: int,
                               arrival_count: int) -> List[int]:
        expected_num_refugees = 10 / 11 * sum(current_caps.values())
        expected_num_cases = expected_num_refugees / self.average_case_size
        std_dev = expected_num_cases / 10

        nbinom_n = expected_num_cases**2 / (std_dev**2 - expected_num_cases)
        nbinom_p = expected_num_cases / std_dev**2

        percentile_cutoff = nbinom.cdf(case_count - 1, nbinom_n, nbinom_p)
        conditioned_percentiles = 1. - uniform.rvs(scale=1. - percentile_cutoff, size=num_trajectories)
        traj_lengths = []
        for tl in nbinom.ppf(conditioned_percentiles, nbinom_n, nbinom_p):
            assert tl >= case_count
            traj_lengths.append(int(tl) - case_count)
        return traj_lengths



class DynamicDualPriceAlgorithm(PriceAlgorithm):
    deterministic = False

    def __init__(self, locality_names, case_pool: CasePool, include_current: bool, which_dual: KindOfDual,
                 trajectory_lengths: TrajectoryLengths, capacities: Capacities, dual_iterations: int = 1,
                 price_multiplier: float = 1.):
        self.case_pool = case_pool
        self.include_current = include_current
        self.which_dual = which_dual
        assert dual_iterations >= 1
        self.trajectory_lengths = trajectory_lengths
        self.capacities = capacities
        self.dual_iterations = dual_iterations

        self.case_counter = 0
        self.arrival_counter = 0

        super().__init__(locality_names, price_multiplier)

    def allocate_batch(self, batch: List[Case], remaining_caps: Dict[str, int], date: datetime.date):
        self.case_counter += len(batch)
        self.arrival_counter += sum(case.size for case in batch)
        return super().allocate_batch(batch, remaining_caps, date)

    def get_prices(self, batch, remaining_caps, date):
        prices = {loc: 0. for loc in self.locality_names}
        case_pool = self.case_pool.get_pool(date)
        full_caps = self.capacities.get_capacities_at_date(date)

        for trajectory_length in self.trajectory_lengths.get_trajectory_lengths(full_caps, self.dual_iterations,
                                                                                self.case_counter,
                                                                                self.arrival_counter):
            trajectory = choices(case_pool, k=trajectory_length)
            if self.include_current:
                trajectory += batch

            new_prices = get_duals(self.locality_names, trajectory, remaining_caps, self.which_dual)
            for loc in self.locality_names:
                prices[loc] += new_prices[loc] / self.dual_iterations

        return prices
