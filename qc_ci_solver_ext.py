#!/home/renke/.conda/envs/forte_env/bin/python
""" 
Update `options` form external file.
originally copied from `/home/renke/papers-collaborative/qc-dsrg/results/qc_1q_nftopt.py`
"""

import numpy as np
# from scipy.optimize import curve_fit
from math import pi
import json
import os
import argparse

from qiskit.providers.aer.noise import NoiseModel
from qiskit import execute

def get_optimal_t(c_x, c_z):
    t = np.arctan(c_x / c_z)
    e_plus  = c_z * np.cos( t) + c_x * np.sin( t)
    e_minus = c_z * np.cos(-t) + c_x * np.sin(-t)
    return t if e_plus <= e_minus else -t


def get_diag_element(occ, scalar_e, oei, tei):
    """
    occ: list of indices for occupied spin orbital
    """
    h = scalar_e
    for i in occ:
        h += oei[(i, i)]
    for i in occ:
        for j in occ:
            h += 0.5 * tei[(i, j, i, j)]
    return h

def intsdict2coeffs(mol_data):
    scalar_energy = mol_data['scalar_energy']['data']
    oei = {(i, j): h_ij for [i, j, h_ij] in mol_data['oei']['data']}
    tei = {(i, j, k, l): h_ijkl
            for [i, j, k, l, h_ijkl] in mol_data['tei']['data']}

    h_00 = get_diag_element([0, 1], scalar_energy, oei, tei)
    h_11 = get_diag_element([2, 3], scalar_energy, oei, tei)
    h_10 = tei[(0, 1, 2, 3)]
    print(f'h_00:{h_00}, h_11:{h_11}, h_10:{h_10}')

    c_0 = (h_00 + h_11) / 2.
    c_z = np.real((h_00 - h_11) / 2.)
    c_x = np.real(h_10)
    
    return c_z, c_x, c_0
    
    
def get_coeffs(r_path, use_dressed_h = False):
    print('---------------------------------------------------------------')
    additional_info = {}
    if use_dressed_h:
        print(f'Read dsrg_ints.json in {r_path}')
        with open(f'{r_path}/dsrg_ints.json', 'r') as read_file:
            mol_data = json.load(read_file)
        c_z, c_x, c_0 = intsdict2coeffs(mol_data)
    else:
        if os.path.isfile(f'{r_path}/forte_ints.json'):
            print(f'Find forte_ints.json in {r_path}')
            with open(f'{r_path}/forte_ints.json', 'r') as read_file:
                mol_data = json.load(read_file)
            c_z, c_x, c_0 = intsdict2coeffs(mol_data)
        
        elif os.path.isfile(f'{r_path}/oq_map.json'):
            print(f'Find oq_map.json in {r_path}')
            with open(f'{r_path}/oq_map.json', "r") as file:
                oq_map_data = json.load(file)

            c_0 = oq_map_data['c0']
            c_z = oq_map_data['cz']
            c_x = oq_map_data['cx']
            additional_info['hamiltonian'] = oq_map_data['hamiltonian']
            additional_info['scalar_e'] = oq_map_data['scalar_e']

    return c_z, c_x, c_0, additional_info


def prepare_1q_var_form(measure_basis, t):
    from qiskit import QuantumCircuit
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


def get_statevec(measure_basis, t):
    from qiskit import Aer
    cir = prepare_1q_var_form(measure_basis, t)
    sim = Aer.get_backend('statevector_simulator')
    statevec = sim.run(cir).result().get_statevector(cir, decimals=9)
    print(f'Statevec: {statevec}')
    return statevec


def measure_1q(measure_basis, t, options):
    """
    Params
    ------
    measure_basis: str
        'Z', 'X', 'Y'
    
    set_backend:  qiskit.providers.aer.backends.qasm_simulator.QasmSimulator, OR:
                qiskit.providers.ibmq.ibmqbackend.IBMQBackend
        eg. set_backend = Aer.get_backend('qasm_simulator')
    
    n_shots: int or float
    
    device_for_noise_model: qiskit.providers.ibmq.ibmqbackend.IBMQBackend,
        eg. device_for_noise_model = provider.get_backend('ibmq_santiago') 
    
    Returns
    -------
    counts: dict
    """
    n_shots = int(options['n_shots'])
    set_backend = options['backend']
    if 'device_for_noise_model' in options.keys():
        device_for_noise_model = options['device_for_noise_model']
        noise_model = NoiseModel.from_backend(device_for_noise_model)
        model_name = device_for_noise_model.name()
    else:
        noise_model = None
        model_name = ''

    if isinstance(t, list) or isinstance(t, np.ndarray) or isinstance(t, tuple):
        params = t[0]
    elif isinstance(t, float):
        params = t
    else:
        print('Parameter Type Not Supported')

    print(f'  t = {params:.6f}, run on {set_backend} {model_name}, measure {measure_basis.upper()} {n_shots} shots.')

    cir = prepare_1q_var_form(measure_basis, params)

    # from qiskit import transpile
    # cir = transpile(cir, set_backend, initial_layout=[3], optimization_level=3)

    # cir_list = []
    # ### num of experiments in the list less than 75
    # if (n_shots > 8192 and n_shots//8192 <= 75):
    #     cir_list.append([cir for i in range(n_shots//8192)])
    # ### Note: the number of experiments supported by the device is 75
    # elif (n_shots//8192 > 75 and (n_shots//8192)%75 > 0):
    #     for _ in range((n_shots//8192)//75):
    #         cirs_75 = [cir for i in range(75)]
    #         cir_list.append(cirs_75)
    #     cir_list.append([cir for i in range((n_shots//8192) % 75)])

    job = execute(cir, shots=n_shots, backend=set_backend, noise_model=noise_model) \
        if set_backend.name() == 'qasm_simulator' else execute(cir, backend=set_backend, shots=n_shots)
    result = job.result()
    counts = result.get_counts()
    for key in ['0', '1']:
        if key not in counts.keys():
            counts[key] = 0

    # counts = {'0': 0, '1': 0}
    # if set_backend.name() == 'qasm_simulator':
    #     job = execute(cir, shots=n_shots, backend=set_backend, noise_model=noise_model)
    #     result = job.result()
    #     exp_counts = result.get_counts()
    #     for key in counts.keys():
    #         if key in exp_counts:
    #             counts[key] += exp_counts[key]
    # elif n_shots <= 8192:
    #     job = execute(cir, shots=n_shots, backend=set_backend)
    #     print(f'    job id: {job.job_id()}')
    #     result = job.result()
    #     exp_counts = result.get_counts()
    #     for key in counts.keys():
    #         if key in exp_counts:
    #             counts[key] += exp_counts[key]
    # else:
    #     ### when n_shots exceeds 8192 limit, use increase-shot trick
    #     ### the number of experiments supported by the device is 75
    #     num_exps = 0
    #     for cirs in cir_list:
    #         num_exps += len(cirs)
    #         job = execute(cirs, shots=8192, backend=set_backend)
    #         print(f'    {len(cirs)*8192} shots, job id: {job.job_id()}')
    #         result = job.result()
    #         for idx in range(len(cirs)):
    #             exp_i_dict = result.get_counts(cirs[idx])
    #             for key in counts.keys():
    #                 if key in exp_i_dict.keys():
    #                     counts[key] += exp_i_dict[key]

    #     remain_shots = n_shots - num_exps*8192
    #     job_remainder = execute(cir, shots=remain_shots, backend=set_backend)
    #     print(f'    {remain_shots} shots, job id: {job_remainder.job_id()}')
    #     result_r = job_remainder.result()
    #     exp_r_counts = result_r.get_counts()
    #     for key in counts.keys():
    #         if key in exp_r_counts:
    #             counts[key] += exp_r_counts[key]

    return counts

def get_amps_1q(counts):
    for key in ['0', '1']:
        if key not in counts.keys():
            counts[key] = 0
    c1sq = np.round(counts['0'] / (counts['0'] + counts['1']), 16)
    c2sq = np.round(counts['1'] / (counts['0'] + counts['1']), 16)
    return (c1sq, c2sq)


def compute_rdms(c1sq, c2sq, avg_x):
    """
    gamma1: list of lists
    gamma2: list of lists
    """
    nso = 4
    g1 = np.zeros([4, 4])

    for i in [0, 1]:
        g1[i, i] = c1sq
    for i in [2, 3]:
        g1[i, i] = c2sq

    # g1 is diagonal, only save non-zero elements
    gamma1 = [[i, i, g1[i,i]] for i in range(nso)]

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

    rdms_dict = {
        'gamma1': {
            'data': gamma1,
            'description': 'one-body density matrix as a list of tuples (i,j,<i^ j>)'
        },
        'gamma2': {
            'data': gamma2,
            'description': 'two-body density matrix as a list of tuples (i,j,k,l,<i^ j^ l k>)',
        }
    }
    return rdms_dict


def get_meas_fitter_object(options):
    ## Import measurement calibration functions
    from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter
    
    n_shots = int(options['n_shots'])
    set_backend = options['backend']
    if 'device_for_noise_model' in options.keys():
        device_for_noise_model = options['device_for_noise_model']
    else:
        device_for_noise_model = None

    print(f'\n   ==> Run Calibration circuits <== ')
    print('n_shots       = %10d' % n_shots)
    print(f'Backend       =      {set_backend}')

    meas_calibs, state_labels = complete_meas_cal(qubit_list=[0], circlabel='mea_cali')
    if set_backend.name() == 'qasm_simulator':
        noise_model = None if device_for_noise_model == None else NoiseModel.from_backend(device_for_noise_model)
        print(f'Noise model   =      {device_for_noise_model.name()}\n{noise_model}')
        job = execute(meas_calibs, shots=n_shots, backend=set_backend, noise_model=noise_model)
    else:
        job = execute(meas_calibs, shots=n_shots, backend=set_backend)

    cal_results = job.result()

    # Calculate the calibration matrix
    fitter = CompleteMeasFitter(cal_results, state_labels, circlabel='mea_cali')
    print(f'Calibration matrix =\n{fitter.cal_matrix}\n')
    meas_filter = fitter.filter

    return meas_filter

def three_pts_quadrature(t_A, options):
    print(f'   ==> Run 3-point Fourier quadrature <==')
    c_z, c_x, _ = options['cz_cx_c0']
    # 3-point Fourier quadrature points (http://arxiv.org/abs/1904.03206)
    # E(t) = a + b*cos(t) + c*sin(t)
    counts_list = [measure_1q('Z', t_A, options),
                   measure_1q('X', t_A, options),

                   measure_1q('Z', t_A+pi/3., options),
                   measure_1q('X', t_A+pi/3., options),

                   measure_1q('Z', t_A-pi/3., options),
                   measure_1q('X', t_A-pi/3., options),
                   ]

    calibration_info = {}
    if 'do_readout_calibration' in options.keys():
        print('\nCalibrate 6 raw counts for 3-point quadrature...')
        calibration_info['3ts_counts_raw'] = counts_list
        # Use filter object to calibrate raw data
        counts_list = [options['filter'].apply(counts) for counts in counts_list.copy()]
        calibration_info['3ts_counts_cal'] = counts_list

    avgs = [get_amps_1q(counts)[0] - get_amps_1q(counts)[1] for counts in counts_list]
    es = [c_z * avgs[2 * i] + c_x * avgs[2 * i + 1] for i in range(3)]

    A = np.matrix([[1, avgs[0], avgs[1]],
                   [1, avgs[2], avgs[3]],
                   [1, avgs[4], avgs[5]]])

    print(f'\nSolve linear equation E(t) = a + b*cos(t) + c*sin(t):')
    a, b, c = np.linalg.solve(A, es)
    print(f'  a = {a}, b = {b}, c = {c}')

    t_opt = get_optimal_t(c, b)
    print(f'\n  t_opt = {t_opt:.9f}\n    <Z>_opt = cos(t_opt) = {np.cos(t_opt):.9f}\n    <X>_opt = sin(t_opt) = {np.sin(t_opt):.9f}')
    print(f'   === End 3-point Fourier quadrature ===\n')
    return t_opt, calibration_info


def run_one_vqe(options):
    c_z, c_x, c_0 = options['cz_cx_c0']
    print(f'c_z = {c_z:.9f}\nc_x = {c_x:.9f}\nc_0 = {c_0:.9f}\n')

    t_A = get_optimal_t(c_x, c_z)
    print(f'  t_A = {t_A:.9f}\n    <Z>_A = cos(t_A) = {np.cos(t_A):.9f}\n    <X>_A = sin(t_A) = {np.sin(t_A):.9f}\n    E_A = {c_0 + c_z*np.cos(t_A) + c_x*np.sin(t_A)}\n')

    if options['skip_3pt_quadrature']:
        print('Skip 3-point Fourier quadrature...\n')
        t_opt = t_A
        calibration_info = {}
    else:
        t_opt, calibration_info = three_pts_quadrature(t_A, options)

    print('Measure Z, X on circuit parametrized by t_opt:')
    counts_z = measure_1q('Z', t_opt, options)
    counts_x = measure_1q('X', t_opt, options)

    def op_averaging(counts_z, counts_x):
        print(f'\nEstimate expectations from counts:')
        (c1sq, c2sq) = get_amps_1q(counts_z)
        avg_z = c1sq - c2sq
        avg_x = get_amps_1q(counts_x)[0] - get_amps_1q(counts_x)[1]
        e = c_z*avg_z + c_x*avg_x + c_0
        print(f'  <Z> = {avg_z}\n  <X> = {avg_x}\n  Energy = {e}')
        return c1sq, c2sq, avg_x, e

    c1sq, c2sq, avg_x, e = op_averaging(counts_z, counts_x)

    if 'do_readout_calibration' in options.keys():
        calibration_info['topt_counts_z_raw'] = counts_z
        calibration_info['topt_counts_x_raw'] = counts_x
        print('\nCalibrate 2 raw counts for final energy...')
        counts_z = options['filter'].apply(counts_z)
        counts_x = options['filter'].apply(counts_x)
        calibration_info['topt_counts_z_cal'] = counts_z
        calibration_info['topt_counts_x_cal'] = counts_x
        c1sq, c2sq, avg_x, e = op_averaging(counts_z, counts_x)

        with open('calibration_info.json', 'w') as file:
            json.dump(calibration_info, file, indent=2)

    print(f'  measure Z: {counts_z})')
    print(f'  measure X: {counts_x})')

    vqe_result = compute_rdms(c1sq, c2sq, avg_x)
    vqe_result['energy'] = {'data': e, 'description': 'energy'}

    vqe_result['t_A'] = t_A
    vqe_result['t_exp'] = t_opt
    vqe_result['c1sq'] = c1sq
    vqe_result['c2sq'] = c2sq
    vqe_result['avg_x'] = avg_x

    return vqe_result


def run_pec(options, i=0):
    print(f'\nSet-{i}')
    
    if 'do_readout_calibration' in options.keys():
        options['filter'] = get_meas_fitter_object(options)

    options['backend_name'] = options['backend'].name().rstrip('simulator').rstrip('_') \
        if 'simulator' in options['backend'].name() \
        else options['backend'].name().lstrip('ibmq').lstrip('_')

    options['noise_model_name'] = '_'+options['device_for_noise_model'].name().lstrip('ibmq').lstrip('_') \
        if ('device_for_noise_model' in options.keys()) else ''
    
    
    rvals = [0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 1.15, 1.2 , \
             1.3 , 1.45, 1.6 , 1.9 , 2.5 , 2.95, 6.  ]

    maindir = options['ints_dir']
    
    for r in rvals:
        r_path = f'{maindir}/{r}'
        c_z, c_x, c_0, additional_info = get_coeffs(r_path,  \
                                                    use_dressed_h=options['use_dressed_h'])
        options['cz_cx_c0'] = (c_z, c_x, c_0)
        
        rdm_path = f'{maindir}/{options["backend_name"]}{options["noise_model_name"]}_{i}/{r}'
        if not os.path.exists(rdm_path):
            os.makedirs(rdm_path)

        os.chdir(rdm_path)
        
        vqe_result = run_one_vqe(options) # save `calibration_info.json` in current working folder.

        with open(f'../vqe.dat', 'a') as file:
            file.write(
                f"{r}    {vqe_result['energy']['data']:.9f}  {vqe_result['t_A']:.7f}  {vqe_result['t_exp']:.7f}  {vqe_result['c1sq']:.9f}  {vqe_result['c2sq']:.9f}  {vqe_result['avg_x']:.9f} \n"
            )
            
        label = f'{options["backend_name"]}{options["noise_model_name"]}'
        with open(f'{r_path}/{label}.dat', 'a') as file:
            file.write(f"{label}_{i}    {vqe_result['energy']['data']}\n")


        rdms = {key: vqe_result[key] for key in ['energy', 'gamma1', 'gamma2']}
        with open(f'rdms.json', 'w') as file:
            json.dump(rdms, file, indent=2)
        print(f'Save rdms.json to {os.getcwd()}')

        os.chdir(maindir)

        if 'hamiltonian' in additional_info.keys():
            evals, _ = np.linalg.eigh(additional_info['hamiltonian'])
            ## Add scalar energies
            evals += [additional_info['scalar_e']] * len(evals)
            e_analytic = -np.sqrt(c_z**2 + c_x**2) + c_0
            print(f'\n   ==> (oq_map.json) Exact results for r = {r} <== ')
            print(f'evals = {evals}\ne_analytic = -np.sqrt(c_z**2 + c_x**2) + c_0 = {e_analytic}')
            print(f'e_analytic - evals[0] = {e_analytic-evals[0]:.12f}\n')


            
if __name__ == "__main__":
    from qiskit import Aer, IBMQ
    IBMQ.load_account()  
    
    options = {  
        'provider_str': 'vqe-dsrg-hardwar' ,
        'backend': None ,
        'n_shots': 20000 ,
        'ints_dir': None ,
        'do_readout_calibration': True  ,  
        'skip_3pt_quadrature': True  ,
        'use_dressed_h': True ,
#         'device_for_noise_model': None  , # name_str
    }
    
    # Update `options` form external file
    options_arr = np.genfromtxt(fname='options.txt', dtype='unicode')
    for line in options_arr:
        key, val = line
        if key in ['do_readout_calibration', 'skip_3pt_quadrature', 'use_dressed_h',]:
            options[key] = bool(options[key])
        else:
            options[key] = val       
    options['n_shots'] = int(options['n_shots'])
   
    
    provider = IBMQ.get_provider(project=options['provider_str'])   
    
    # change name string to backend object
    backend = Aer.get_backend('qasm_simulator') if options['backend'] == 'qasm_simulator' else provider.get_backend(options['backend']) 
    options['backend'] = backend
    
    if 'device_for_noise_model' in options.keys():
        if options['device_for_noise_model'] != None:
            # change name string to backend object
            options['device_for_noise_model'] = provider.get_backend(options['device_for_noise_model'])       
    print(f'options=\n{options}') 
    
    run_pec(options)
                                                                     
#     for i in range(1, 9): 
#         run_pec(options, i)