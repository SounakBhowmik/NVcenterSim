from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from nvquantum import NVBackend


@dataclass
class RunOutputs:
    density_matrix: np.ndarray  # 2x2 complex
    counts: Dict[str, int]


class RamseySimulator:
    """
    Runs circuits through AerSimulator with NVQuantum noise model.
    IMPORTANT: We transpile against NVBackend.target, then inject save_density_matrix
    AFTER transpilation (because save_density_matrix isn't in the target gate set).
    """
    def __init__(self, nv_backend: NVBackend, optimization_level: int = 2):
        self.nv_backend = nv_backend
        self.noise_model = nv_backend.noise_model
        self.target = nv_backend.target
        self.optimization_level = optimization_level

        # density_matrix method required for density matrix snapshots under noise
        self.aer = AerSimulator(method="density_matrix")

    def _insert_density_snapshot(self, tqc: QuantumCircuit, label: str = "rho") -> None:
        # Find first measurement instruction index
        first_meas_idx = None
        for idx, inst in enumerate(tqc.data):
            if inst.operation.name == "measure":
                first_meas_idx = idx
                break
        if first_meas_idx is None:
            raise RuntimeError("No measurement found in transpiled circuit; cannot insert snapshot.")

        # Create a snapshot instruction and insert before the first measure
        snap = QuantumCircuit(tqc.num_qubits)
        try:
            snap.save_density_matrix(label=label)
        except Exception:
            snap.save_density_matrix()
        tqc.data.insert(first_meas_idx, snap.data[0])

    def run(self, qc: QuantumCircuit, shots: int, dm_label: str = "rho") -> RunOutputs:
        # 1) Transpile physical circuit to NV target
        tqc = transpile(
            qc,
            backend=self.aer,
            optimization_level=self.optimization_level,
            target=self.target,
        )

        # 2) Inject density snapshot AFTER transpile
        self._insert_density_snapshot(tqc, label=dm_label)

        # 3) Run
        job = self.aer.run(tqc, shots=shots, noise_model=self.noise_model)
        result = job.result()

        counts = result.get_counts(0)
        data0 = result.data(0)

        # 4) Extract density matrix
        if dm_label in data0:
            rho = data0[dm_label]
        elif "density_matrix" in data0:
            rho = data0["density_matrix"]
        else:
            raise RuntimeError(f"Density matrix not found. Keys: {list(data0.keys())}")

        return RunOutputs(density_matrix=np.array(rho, dtype=np.complex128), counts=counts)
