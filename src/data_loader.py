"""
src/data_loader.py
==================
Data loading utilities for the cybersecurity application notebook.

Two data sources are supported:

synthesize_cyber_attacks(T_hours, rng)
    Generates a synthetic server attack log mixing two realistic attack
    profiles: broad-spectrum botnet scanning and targeted APT probing.

load_unsw_nb15(csv_path, max_events)
    Loads the raw UNSW-NB15 network traffic CSV and extracts the attack
    timestamp sequence.  Column indices follow the official UNSW-NB15
    data description document (no header row).
"""

import numpy as np
import pandas as pd
from .simulator import simulate_hawkes


def synthesize_cyber_attacks(T_hours=12, rng=None):
    """
    Generate a realistic synthetic server attack log.

    Mixes two independent Hawkes processes representing two empirically
    distinct attack profiles observed in production firewalls:

    Profile A — Botnet scanning:
        High background rate (~50 hits/hour), very weak self-excitation
        (eta = alpha/beta = 0.10).  Models automated broad-spectrum port
        scans that arrive at roughly constant rate.

    Profile B — APT lateral movement:
        Very low background rate (~2 hits/hour), strong self-excitation
        (eta = 0.43).  Models targeted exploit sequences where each
        successful probe triggers a rapid burst of follow-on probes.

    Parameters
    ----------
    T_hours : float
        Observation window length in hours.
    rng : numpy.random.Generator or None
        Random number generator.  Pass a seeded generator for
        reproducibility.

    Returns
    -------
    events    : ndarray — merged, sorted attack timestamps in seconds.
    labels    : ndarray of int — ground-truth label per event
                (0 = botnet, 1 = APT).
    T_seconds : float — total observation horizon in seconds.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    T_seconds = T_hours * 3600

    mu_bot, alpha_bot, beta_bot = 50.0 / 3600, 0.05, 0.5
    botnet_events = simulate_hawkes(mu_bot, alpha_bot, beta_bot,
                                    T_seconds, rng)

    mu_apt, alpha_apt, beta_apt = 2.0 / 3600, 0.85, 2.0
    apt_events = simulate_hawkes(mu_apt, alpha_apt, beta_apt,
                                 T_seconds, rng)

    all_events = np.concatenate([botnet_events, apt_events])
    labels     = np.concatenate([
        np.zeros(len(botnet_events), dtype=int),
        np.ones(len(apt_events),     dtype=int),
    ])
    order = np.argsort(all_events)
    return all_events[order], labels[order], T_seconds


def load_unsw_nb15(csv_path, max_events=2000):
    """
    Load the raw UNSW-NB15 network traffic log and extract the attack
    timestamp sequence.

    The raw CSV files (UNSW-NB15_1.csv through _4.csv) carry no header
    row.  Column indices are taken from the official UNSW-NB15 feature
    description document:
        Column 28 (0-indexed) : Stime  — connection start time (Unix epoch).
        Column 48 (0-indexed) : Label  — 0 = normal traffic, 1 = attack.

    Only attack records (Label == 1) are retained, and timestamps are
    normalised to begin at t = 0.

    Parameters
    ----------
    csv_path   : str — path to one of the raw UNSW-NB15 CSV files.
    max_events : int — cap on the number of attack events loaded
                 (prevents excessive memory use on large files).

    Returns
    -------
    timestamps : ndarray — attack event times in seconds from t = 0.
    labels     : ndarray of int — all ones (attack); included for
                 API consistency with synthesize_cyber_attacks.
    T_total    : float — observation horizon (last timestamp + 1).
    """
    df = pd.read_csv(
        csv_path,
        header=None,
        usecols=[28, 48],
        names=['Stime', 'Label'],
        low_memory=False,
    )

    df = df.sort_values('Stime')
    df_attacks = df[df['Label'] == 1].head(max_events).copy()

    timestamps = df_attacks['Stime'].astype(float).values
    timestamps = timestamps - timestamps[0]

    labels  = np.ones(len(timestamps), dtype=int)
    T_total = float(timestamps[-1]) + 1.0
    return timestamps, labels, T_total