import numpy as np
import pandas as pd
from collections import defaultdict

class CpSIRSimulator:
    """
    Minimal cpSI-R simulator.
    - nodes can be in states: S (susceptible), I (infected/active), R (inactive)
    - infected nodes attempt to infect neighbors for up to tau seconds since their last attempt.
    - repeated exposures increase infection probability via reinforcement parameter alpha.
    - reactivation: an inactive node can be reactivated if contacted by an active node.
    This implementation is simple and meant for clarity and reproducibility.
    """
    def __init__(self, edges_df, lambda_p=0.03, mu=0.01, alpha=0.5, tau=300):
        # edges_df: pandas DataFrame with columns src,tgt,timestamp (timestamps assumed sorted)
        self.edges = edges_df.copy().reset_index(drop=True)
        self.lambda_p = lambda_p
        self.mu = mu
        self.alpha = alpha
        self.tau = tau
        self.nodes = set(self.edges['src']).union(set(self.edges['tgt']))
        # adjacency per time: {t: [(u,v), ...]}
        self.time_index = defaultdict(list)
        for r in self.edges.itertuples():
            self.time_index[r.timestamp].append((int(r.src), int(r.tgt)))
        self.times = sorted(self.time_index.keys())
    def run(self, seed_set, max_time=None, random_state=None):
        """
        Run one stochastic realization of cpSI-R.
        seed_set: iterable of initial infected nodes
        returns: dict of final states and infection fraction
        """
        rng = np.random.default_rng(random_state)
        # states: 0=S,1=I,2=R
        state = {n:0 for n in self.nodes}
        last_active_time = {n: -np.inf for n in self.nodes}
        exposure_count = {n:0 for n in self.nodes}
        for s in seed_set:
            if s in state:
                state[s] = 1
                last_active_time[s] = -1  # active from start
        max_time = max_time if max_time is not None else (max(self.times)+self.tau)
        for t in self.times:
            if t>max_time:
                break
            # process edges at time t
            for (u,v) in self.time_index[t]:
                # if u is active (I), it may try to infect v
                if state.get(u,0)==1:
                    # check u still within tau since last_active_time for attempts
                    if (t - last_active_time.get(u, -1)) <= self.tau:
                        # compute dynamic infection probability: base * (1 - exp(-alpha * exposures_to_v_from_u))
                        exposure_count[(u,v)] = exposure_count.get((u,v),0) + 1
                        k = exposure_count[(u,v)]
                        p = self.lambda_p * (1 - np.exp(-self.alpha * k))
                        if rng.random() < p:
                            if state.get(v,0)==0:
                                state[v] = 1
                                last_active_time[v] = t
                            elif state.get(v,0)==2:
                                # reactivation
                                state[v] = 1
                                last_active_time[v] = t
                # simple symmetric handling (if v->u exists as separate edge events, it will be handled)
            # recovery step: infected nodes become inactive with prob mu
            for n in list(self.nodes):
                if state[n]==1:
                    if rng.random() < self.mu:
                        state[n]=2
            # Note: this minimal simulator does not model continuous-time inter-event waiting; it follows discrete event timestamps
        # compute final stats
        counts = {0:0,1:0,2:0}
        for n,s in state.items():
            counts[s]+=1
        total = len(state)
        infected_fraction = (counts[1]+counts[2]) / total
        return {'counts':counts, 'infected_fraction':infected_fraction, 'state':state}
