# qc-dsrg

1. Intergals and mapping: 
- [forte_get_1q_coeffs.py](https://github.com/RenkeHuang/qc-dsrg/blob/main/forte_get_1q_coeffs.py) uses [Forte](https://github.com/evangelistalab/forte) `master` branch to compute one-qubit mapping coefficients using as_ints object, avoiding `qc` branch and *external* option for active_space_solver to get integrals.
- Alternatively, see `$HOME/computations/H2/cas/CAS_INP`, use `write_as_h` option to get as_ham, and run `$HOME/computations/H2/cas/get_oq_map.py` to generate `oq_map.json`

2. Experiments on ibmq backend
- [qc_ci_solver.py](https://github.com/RenkeHuang/qc-dsrg/blob/main/qc_ci_solver.py)
- [Qiskit](https://github.com/Qiskit/qiskit) is installed in **forte_env** conda enviroment.
- copies of `qc_ci_solver.py` on the cluster:
    /home/renke/computations/H2/cas/cc-pV5Z/qc_ci_solver.py 
    /home/renke/computations/bicbut_isomerization/qc_ci_solver.py 

3. SA-MR-LDSRG(2) downfolding
- `qc`branch, use *external* active_space_solver
- use `relax_ref  once` option to get `dsrg_ints.json` (1-QDSRG or 2-QDSRG approximated), and `external_partial_relax  true` get the exact relaxed energy of 1-QDSRG or 2-QDSRG.

