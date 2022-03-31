#!/home/renke/.conda/envs/psi4-2021/bin/python
"""
originally copied from ~/papers-collaborative/qc-dsrg/results/qc_1q_nftopt.py
"""

import numpy as np
# from scipy.optimize import curve_fit
from math import pi
import json
import os

from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.providers.aer.noise import NoiseModel
## Import measurement calibration functions
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter

IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')


class One_qubit_ci_solver:
    def __init__(self, c_dict):
        self.c0 = c_dict['c0']
        self.cz = c_dict['cz']
        self.cx = c_dict['cx']
        self.h_tuple = c_dict['h_tuple']

        self.e_analytic = -np.sqrt(cz**2 + cx**2) + c0
        self.t_analytic = self.optimal_t(np.arctan(c_x/c_z))


    def optimal_t(self, t):
        e_plus = cz*np.cos(t) + cx*np.sin(t)
        e_minus = cz*np.cos(-t) + cx*np.sin(-t)
        if e_plus <= e_minus: return t else -t
    
    
    def get_amps_1q(self, counts):
        for key in ['0', '1']:
            if key not in counts.keys():
                counts[key] = 0
        c1sq = np.round(counts['0']/(counts['0'] + counts['1']), 16)
        c2sq = np.round(counts['1']/(counts['0'] + counts['1']), 16)
        return (c1sq, c2sq)
    

    def prepare_1q_var_form(self, measure_basis, t):
        cir = QuantumCircuit(1, 1)
        cir.ry(t, 0)

        if measure_basis.upper() == 'Z':
            cir.measure(0, 0)
        elif measure_basis.upper() == 'X':
            cir.h(0)
            cir.measure(0, 0)
        elif measure_basis.upper() == 'Y':
            ### u2(φ,λ) = u(π/2, φ, λ)
            cir.u(pi/2, -pi/2, pi/2, 0)
            cir.measure(0, 0)
        else:
            print('Type error')

        return cir

    def get_statevec_1q(self, measure_basis, t):
        simulator = Aer.get_backend('statevector_simulator')
        cir = self.prepare_1q_var_form(t, measure_basis=measure_basis)
        statevec = execute(cir, backend=simulator).result().get_statevector()
        return statevec

    def measure_1q(self, measure_basis, t, set_backend, n_shots=8192, device_for_noise_model=None):
        """
        Params
        ------
        measure_basis: str
            'Z', 'X', 'Y'
        
        set_backend:  qiskit.providers.aer.backends.qasm_simulator.QasmSimulator, OR:
                    qiskit.providers.ibmq.ibmqbackend.IBMQBackend
            eg. set_backend = Aer.get_backend('qasm_simulator')
        
        n_shots: int or float, default: 8192
        
        device_for_noise_model: qiskit.providers.ibmq.ibmqbackend.IBMQBackend,
            eg. device_for_noise_model = provider.get_backend('ibmq_santiago') 
        
        
        Returns
        -------
        counts: dict
        """
        n_shots = int(n_shots)

        if isinstance(t, list) or isinstance(t, np.ndarray):
            params = t[0]
        elif isinstance(t, float):
            params = t
        else:
            print('Input Type Not Supported')

        noise_model = None if device_for_noise_model == None else NoiseModel.from_backend(device_for_noise_model)
        model_name = None if device_for_noise_model == None else device_for_noise_model.name()

        if set_backend.name() == 'qasm_simulator':
            print(f't = {t}, run on {set_backend}({model_name}), measure {measure_basis.upper()} {n_shots} shots.')
        else:
            print(f't = {t}, run on {set_backend}, measure {measure_basis.upper()} {n_shots} shots.')

        cir = self.prepare_1q_var_form(params, measure_basis=measure_basis)

        cir_list = []
        if (n_shots > 8192 and n_shots//8192 <= 75):
            cir_list.append([cir for i in range(n_shots//8192)])
        ### Note: the number of experiments supported by the device is 75
        elif (n_shots//8192 > 75 and (n_shots//8192)%75 > 0):
            for k in range((n_shots//8192)//75):
                cirs_75 = [cir for i in range(75)]
                cir_list.append(cirs_75)
            cir_list.append([cir for i in range((n_shots//8192) % 75)])


        counts = {'0':0, '1':0}

        if set_backend.name() == 'qasm_simulator':
            job = execute(cir, backend=set_backend, shots=n_shots, noise_model=noise_model)
            result = job.result()
            exp_counts = result.get_counts()
            for key in counts.keys():
                if key in exp_counts:
                    counts[key] += exp_counts[key]
        elif n_shots <= 8192:
            job = execute(cir, backend=set_backend, shots=n_shots)
            print(f'    job id: {job.job_id()}')
            result = job.result()
            exp_counts = result.get_counts()
            for key in counts.keys():
                if key in exp_counts:
                    counts[key] += exp_counts[key]
        else:
            ### when n_shots exceeds 8192 limit, use increase-shot trick
            ### the number of experiments supported by the device is 75
            num_exps = 0
            for cirs in cir_list:
                num_exps += len(cirs)
                job = execute(cirs, backend=set_backend, shots=8192)
                print(f'    {len(cirs)*8192} shots, job id: {job.job_id()}')
                result = job.result()
                for idx in range(len(cirs)):
                    exp_i_dict = result.get_counts(cirs[idx])
                    for key in counts.keys():
                        if key in exp_i_dict.keys():
                            counts[key] += exp_i_dict[key]

            remain_shots = n_shots - num_exps*8192
            job_remainder = execute(cir, backend=set_backend, shots=remain_shots)
            print(f'    {remain_shots} shots, job id: {job_remainder.job_id()}')
            result_r = job_remainder.result()
            exp_r_counts = result_r.get_counts()
            for key in counts.keys():
                if key in exp_r_counts:
                    counts[key] += exp_r_counts[key]

        return counts





def compute_rdms(counts_z, counts_x, *params):
    """
    Return
    ------
    e: float
    gamma1: list of lists
    gamma2: list of lists
    """
    c_z, c_x, shifted_e, r, key = params
    print(f'\nComputing RDMs for {key} data:')
    print(f'    counts_z = {counts_z},\n    counts_x = {counts_x}')

    (c1sq, c2sq) = get_amps_1q(counts_z)
    avg_z = c1sq - c2sq

    avg_x = get_amps_1q(counts_x)[0] - get_amps_1q(counts_x)[1]

    print(f'    avg_z = {avg_z}, avg_x = {avg_x}')

    e = c_z*avg_z + c_x*avg_x + shifted_e
    print(f'    meas_e = {c_z*avg_z + c_x*avg_x} (total_e (add scalar) = {e})')

    g1_dict = {}
    for a1 in [0,2]:
        for a2 in [0,2]:
            g1_dict[(a1,a2)] = 0.0
    for b1 in [1,3]:
        for b2 in [1,3]:
            g1_dict[(b1,b2)] = 0.0
    g1_dict[(0,0)] = g1_dict[(1,1)] = c1sq
    g1_dict[(2,2)] = g1_dict[(3,3)] = c2sq
    gamma1 = [[i, j, g1_dict[(i,j)]] for (i,j) in g1_dict.keys()]
    # print(f'\ngamma1 =\n{gamma1}')

    g2_dict = {}
    g2_dict[(0,1,0,1)] = g2_dict[(1,0,1,0)] =  c1sq
    g2_dict[(1,0,0,1)] = g2_dict[(0,1,1,0)] = -c1sq
    g2_dict[(2,3,2,3)] = g2_dict[(3,2,3,2)] =  c2sq
    g2_dict[(3,2,2,3)] = g2_dict[(2,3,3,2)] = -c2sq

    avg_01 = avg_10 = avg_x/2.
    g2_dict[(0,1,2,3)] = g2_dict[(1,0,3,2)] =  avg_01
    g2_dict[(1,0,2,3)] = g2_dict[(0,1,3,2)] = -avg_01
    g2_dict[(2,3,0,1)] = g2_dict[(3,2,1,0)] =  avg_10
    g2_dict[(3,2,0,1)] = g2_dict[(2,3,1,0)] = -avg_10

    gamma2 = [[i, j, k, l, g2_dict[(i,j,k,l)]] for (i,j,k,l) in g2_dict.keys()]
    # print(f'\ngamma2 =\n{gamma2}\n')

    with open(f'./meas_{key}.dat', 'a') as file:
        file.write(
            f'{r}    {c1sq}    {c2sq}    {avg_01}    {c_z}    {c_x}    {shifted_e}\n')

    return e, gamma1, gamma2


def run_single_r(r, ints_path, n_shots, set_backend, device_for_noise_model=None):
    """
    Return
    ------
    rdms_dict: list of 2 dict (raw, calibrated)
    """

    print('\n\n=================')
    print(f'r = {r} angstrom')
    print('-----------------')

    ### Read integrals and get qubit hamiltonian
    with open(ints_path, "r") as read_file:
        mol_data = json.load(read_file)
    scalar_energy = mol_data['scalar_energy']['data']
    oei = {(i,j):h_ij for [i,j,h_ij] in mol_data['oei']['data']}
    tei = {(i,j,k,l):h_ijkl for [i,j,k,l,h_ijkl] in mol_data['tei']['data']}

    h_00 = get_diag_element([0,1], scalar_energy, oei, tei)
    h_11 = get_diag_element([2,3], scalar_energy, oei, tei)
    h_10 = tei[(0,1,2,3)]

    c_z = np.real((h_00 - h_11)/2.)
    c_x = np.real(h_10)
    print(f'c_z = {c_z}\nc_x = {c_x}')
    shifted_e = (h_00+h_11)/2.
    print(f'shifted_e = {shifted_e}')

    print(f'e_theory = {-np.sqrt(c_z**2+c_x**2)+shifted_e}')

    print('\nQiskit experiments')
    print('------------------')

    ### Get the analytical solution for the optimal angle
    t_A = np.arctan(c_x/c_z)

    ### Prepare for 3-point analytical tomography fitting
    raw_counts_list = [measure_1q('Z', t_A, set_backend, n_shots=n_shots, device_for_noise_model=device_for_noise_model),
                       measure_1q('X', t_A, set_backend, n_shots=n_shots, device_for_noise_model=device_for_noise_model),

                       measure_1q('Z', t_A+pi/3., set_backend, n_shots=n_shots, device_for_noise_model=device_for_noise_model),
                       measure_1q('X', t_A+pi/3., set_backend, n_shots=n_shots, device_for_noise_model=device_for_noise_model),

                       measure_1q('Z', t_A-pi/3., set_backend, n_shots=n_shots, device_for_noise_model=device_for_noise_model),
                       measure_1q('X', t_A-pi/3., set_backend, n_shots=n_shots, device_for_noise_model=device_for_noise_model),
                       ]

    ### Measurement Error Mitigation
    meas_calibs, state_labels = complete_meas_cal(qubit_list=[0], circlabel='mea_cali')

    if set_backend.name() == 'qasm_simulator':
        noise_model = None if device_for_noise_model == None else NoiseModel.from_backend(device_for_noise_model)
        print(f'Noise model for calibration (if QASM_Sim): {noise_model}')
        job = execute(meas_calibs, backend=set_backend, shots=n_shots, noise_model=noise_model)
    else:
        job = execute(meas_calibs, backend=set_backend, shots=n_shots)

    cal_results = job.result()

    # Calculate the calibration matrix
    meas_fitter = CompleteMeasFitter(cal_results, state_labels, circlabel='mea_cali' )
    print(f'Calibration matrix =\n{meas_fitter.cal_matrix}\n')

    # Get the filter object to calibrate raw data
    meas_filter = meas_fitter.filter
    calibrated_counts_list = [meas_filter.apply(counts) for counts in raw_counts_list]

    three_point_cts_dict = {'raw':{}, 'cal':{}}
    three_point_cts_dict['raw']['t_points'] = [t_A, t_A+pi/3., t_A-pi/3.]
    three_point_cts_dict['raw']['counts_list'] = raw_counts_list
    three_point_cts_dict['cal']['counts_list'] = calibrated_counts_list

    final_cts_dict = {'raw': {}, 'cal': {}}
    # 3-point Fourier quadrature points (http://arxiv.org/abs/1904.03206)
    # E(t) = a + b*cos(t) + c*sin(t)
    for key in three_point_cts_dict.keys():
        print(f'\n{key} data:')
        counts_list = three_point_cts_dict[key]['counts_list']

        avgs = [get_amps_1q(counts)[0] - get_amps_1q(counts)[1]  \
                for counts in counts_list]
        three_point_cts_dict[key]['avgs'] = avgs

        es = [c_z*avgs[2*i] + c_x*avgs[2*i+1] for i in range(3)]
        three_point_cts_dict[key]['es'] = es

        print(f'    avgs_{key} = {avgs}, \n    es_{key} = {es}')

        A = np.matrix([[1, avgs[0], avgs[1]],
                       [1, avgs[2], avgs[3]],
                       [1, avgs[4], avgs[5]]])

        coeffs = np.linalg.solve(A, es)
        three_point_cts_dict[key]['abc_tuple'] = tuple(coeffs)

        a, b, c = coeffs
        t = np.arctan(c/b)
        print(f'    a = {a}, b = {b}, c = {c}, t_{key} = {t}')

        final_cts_dict[key]['t_optimal'] = t
        final_z_x_counts = \
            [measure_1q('Z', t, set_backend, n_shots=n_shots, device_for_noise_model=device_for_noise_model),
             measure_1q('X', t, set_backend, n_shots=n_shots, device_for_noise_model=device_for_noise_model)]

        final_cts_dict[key]['counts_list'] = [
            meas_filter.apply(counts) for counts in final_z_x_counts] if key == 'cal' else final_z_x_counts

    nft_log = {'3_point_info': three_point_cts_dict,
               'final_t_info': final_cts_dict}

    # def e_t(t, a, b, c):
    #     return a + b*np.cos(t) + c*np.sin(t)
    # ts = [t_A, t_A+pi/3., t_A-pi/3.]
    # abc_raw, abc_raw_cov = curve_fit(e_t, ts, es_raw)
    # abc_cal, abc_cal_cov = curve_fit(e_t, ts, es_cal)

    rdms_dict = {}
    for key in final_cts_dict.keys():
        counts_list = final_cts_dict[key]['counts_list']
        e, gamma1, gamma2 = compute_rdms(*counts_list, c_z, c_x, shifted_e, r, key)

        rdms = {
            'energy': {'data': e, 'description': 'energy'},
            'gamma1': {
                'data': gamma1,
                'description': 'one-body density matrix as a list of tuples (i,j,<i^ j>)'
            },
            'gamma2': {
                'data': gamma2,
                'description': 'two-body density matrix as a list of tuples (i,j,k,l,<i^ j^ l k>)',
            }
        }
        rdms_dict[key] = rdms

    return rdms_dict, nft_log

def run_pec(i=1, wkdir= os.getcwd()):
    print(f'\nrep {i}\n')

    """1. Get backend(s)"""
    # aer_statevec_sim = Aer.get_backend('statevector_simulator')
    qasm_sim_aer = Aer.get_backend('qasm_simulator')
    # ibmq_qasm_sim = provider.get_backend('ibmq_qasm_simulator')
    ### QV32
    # device_santiago = provider.get_backend('ibmq_santiago')
    # device_manila = provider.get_backend('ibmq_manila')
    # device_bogota = provider.get_backend('ibmq_bogota')  #
    ### QV16
    # device_quito = provider.get_backend('ibmq_quito')  #
    device_belem = provider.get_backend('ibmq_belem')  #
    ### QV8
    # device_lima = provider.get_backend('ibmq_lima')
    ### 1 qubit
    # device_armonk = provider.get_backend('ibmq_armonk')

    ###### Device:
    set_backend = device_belem
    device_for_noise_model = None
    ###### QASM w/ wo/ noise
    # set_backend = qasm_sim_aer
    # device_for_noise_model = None
    ### build noise model from backend properties
    # device_for_noise_model = device_belem
    # print(NoiseModel.from_backend(device_belem))
    ######

    n_shots = 8192

    device_tag = set_backend.name().rstrip('simulator').rstrip('_') if 'simulator' in set_backend.name() \
        else set_backend.name().lstrip('ibmq').lstrip('_')

    noise_model_tag = '-'+device_for_noise_model.name().lstrip('ibmq').lstrip('_') if device_for_noise_model != None else ''

    #
    rvals = [ 0.6, 0.65, 0.7 , 0.75, 0.8, 0.85, 1.15, 1.2, 1.3 , 1.45, 1.6 , 1.9, 2.5 , 2.95, 6.]

    # rdm_path_name = f'{wkdir}/3z_qasm_1e{int(np.log10(n_shots))}'
    # rdm_path_name = f'{wkdir}/3z_qasm_{n_shots}_{i}'
    rdm_path_name = f'{wkdir}/3z_{device_tag}{noise_model_tag}_{n_shots}_{i}'


    for r in rvals:
        ints_path = f'{wkdir}/ints_tz/{r}/forte_ints.json'

        rdms_dict, nft_log = run_single_r(
            r, ints_path, n_shots, set_backend, device_for_noise_model=device_for_noise_model)

        nft_log_folder = f'{rdm_path_name}/OPTLOG'
        if not os.path.exists(nft_log_folder):
            os.makedirs(nft_log_folder)
        os.chdir(nft_log_folder)
        with open(f'r_{r}.json', 'w') as info_file:
            json.dump(nft_log, info_file, indent=4)
        os.chdir(wkdir)

        for key in rdms_dict.keys():
            rdms = rdms_dict[key]

            r_folder = f'{rdm_path_name}/{key}/r_{r}'

            if not os.path.exists(r_folder):
                os.makedirs(r_folder)

            e_vqe = rdms_dict[key]['energy']['data']
            with open(f'{rdm_path_name}/{key}/pec.dat', 'a') as file:
                file.write(f'{r}     {e_vqe}\n')

            os.chdir(r_folder)

            with open('rdms.json', 'w') as file:
                json.dump(rdms, file, indent=4)

            os.chdir(wkdir)


if __name__ == "__main__":
    for i in range(1, 6):
        run_pec(i)
