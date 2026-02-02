from __future__ import annotations
from typing import Dict, List
from nvquantum import NVBackend


def make_single_nv_memory(t1_s: float, t2_s: float, readout_err: float) -> List[Dict]:
    """Deterministic single-qubit nv_memory entry compatible with NVBackend."""
    return [{
        "x": 0.0,
        "y": 0.0,
        "intensity": 150.0,
        "qubit_id": 0,
        "T1": float(t1_s),
        "T2": float(t2_s),
        "readout_err": float(readout_err),
    }]


def build_nv_backend(t1_s: float, t2_s: float, readout_err: float) -> NVBackend:
    """Create a 1-qubit NVBackend from NVQuantum."""
    nv_memory = make_single_nv_memory(t1_s, t2_s, readout_err)
    return NVBackend(nv_memory=nv_memory, max_qubits=1)
