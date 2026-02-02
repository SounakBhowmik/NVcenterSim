from __future__ import annotations
import math
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from nvquantum import gamma_e  # Hz/T


def phase_from_B(B_tesla: float, t_seconds: float) -> float:
    """phi = 2*pi*gamma_e*B*t"""
    return 2.0 * math.pi * float(gamma_e) * float(B_tesla) * float(t_seconds)


def build_ramsey_circuit(phi: float) -> QuantumCircuit:
    """
    Physical Ramsey circuit only (no save_density_matrix here, to keep transpiler happy):
      |0> --H-- Rz(phi) --H-- measure
    """
    q = QuantumRegister(1, "q")
    c = ClassicalRegister(1, "c")
    qc = QuantumCircuit(q, c)
    qc.h(q[0])
    qc.rz(phi, q[0])
    qc.h(q[0])
    qc.measure(q[0], c[0])
    return qc
