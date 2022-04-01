# qc-dsrg

1. Intergals and mapping: 
- [forte_get_1q_coeffs.py](https://github.com/RenkeHuang/qc-dsrg/blob/main/forte_get_1q_coeffs.py) uses [Forte](https://github.com/evangelistalab/forte) `master` branch to compute one-qubit mapping coefficients using as_ints object, avoiding `qc`branch and *external* active_space_solver to get integrals.

2. Experiments on ibmq backend
- [qc_ci_solver.py](https://github.com/RenkeHuang/qc-dsrg/blob/main/qc_ci_solver.py)
- [Qiskit](https://github.com/Qiskit/qiskit) is installed in **psi4-2021** conda enviroment.

3. SA-MR-LDSRG(2) downfolding
- `qc`branch, need to use *external* active_space_solver

