from __future__ import annotations
import math
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from nvquantum import gamma_e  # Hz/T


def phase_from_B(B_tesla: float, t_seconds: float) -> float:
    """phi = 2*pi*gamma_e*B*t"""
    return 2.0 * math.pi * float(gamma_e) * float(B_tesla) * float(t_seconds)

def build_ramsey_circuit(phi: float, meas_basis: str = "Z") -> QuantumCircuit:
    """
    1-qubit Ramsey-like circuit:
      |0> --H-- Rz(phi) --H-- [basis change] --measure

    meas_basis:
      - "Z": measure in computational basis
      - "X": apply H then measure
      - "Y": apply Sdg then H then measure (so Z-measure corresponds to Y)
    """
    meas_basis = meas_basis.upper()
    if meas_basis not in ("X", "Y", "Z"):
        raise ValueError(f"Invalid meas_basis={meas_basis}. Use 'X','Y','Z'.")

    q = QuantumRegister(1, "q")
    c = ClassicalRegister(1, "c")
    qc = QuantumCircuit(q, c)

    qc.h(q[0])
    qc.rz(phi, q[0])
    qc.h(q[0])

    if meas_basis == "X":
        qc.h(q[0])
    elif meas_basis == "Y":
        qc.sdg(q[0])
        qc.h(q[0])

    qc.measure(q[0], c[0])
    return qc