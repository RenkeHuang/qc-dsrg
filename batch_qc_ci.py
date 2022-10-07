#!/home/renke/.conda/envs/forte_env/bin/python
""" 
submit one job to a device with list of circuits [calib_cir1, calib_cir2, cir_z, cir_x,...]
"""

import numpy as np
from math import pi
import json
import os

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


def get_coeffs(ints_path, use_dressed_h = True):
    print('---------------------------------------------------------------')
    additional_info = {}
    if use_dressed_h:
        print(f'Read dsrg_ints.json in {ints_path}')
        with open(f'{ints_path}/dsrg_ints.json', 'r') as read_file:
            mol_data = json.load(read_file)
        c_z, c_x, c_0 = intsdict2coeffs(mol_data)

    else:
        if os.path.isfile(f'{ints_path}/forte_ints.json'):
            print(f'Find forte_ints.json in {ints_path}')
            with open(f'{ints_path}/forte_ints.json', 'r') as read_file:
                mol_data = json.load(read_file)
            c_z, c_x, c_0 = intsdict2coeffs(mol_data)

        elif os.path.isfile(f'{ints_path}/oq_map.json'):
            print(f'Find oq_map.json in {ints_path}')
            with open(f'{ints_path}/oq_map.json', "r") as file:
                oq_map_data = json.load(file)

            c_0 = oq_map_data['c0']
            c_z = oq_map_data['cz']
            c_x = oq_map_data['cx']
            additional_info['hamiltonian'] = oq_map_data['hamiltonian']
            additional_info['scalar_e'] = oq_map_data['scalar_e']
        else:
            print('Error: integral files not found!')

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



def my_oq_calibrate(calcirs_counts, raw_counts):
    identity_cir_cts, x_cir_cts = calcirs_counts
    c00, c10 = get_amps_1q(identity_cir_cts)
    c01, c11 = get_amps_1q(x_cir_cts)
    
    cal_mat = np.array([[c00, c01],
                        [c10, c11]])
    print(f'Calibration matrix =\n{cal_mat}')
    if isinstance(raw_counts, dict):
        raw_cts = [np.array([raw_counts['0'], raw_counts['1']]), ]
    elif isinstance(raw_counts, list):
        raw_cts = [np.array([counts['0'], counts['1']]) for counts in raw_counts]
    else:
        print('Type for `raw counts` not supported.')
    
    cal_counts_vec = [np.dot(np.linalg.inv(cal_mat), vec) for vec in raw_cts]
    cal_counts = [{'0': vec[0], '1': vec[1]} for vec in cal_counts_vec]
    return cal_counts


def run_batch_qc(options, topt_dict, nsets, restart_from_cal_counts=False):
    maindir = options['maindir']
    ints_type = options['ints_type']
    l3 = 'l3_' if 'l3' in f"{ints_type.lstrip('2-qldsrg').replace('/3pdc0', '_l3')}" else ''
    label = f"{options['backend_name']}{options['noise_model_name']}"  
    
    if restart_from_cal_counts: 
        print(f'\n   ==> Restart from {label}_cal_cts.json <== ')  
        print(f'Compute RDM       = {options["compute_rdms"]}')
        with open(f'{maindir}/{l3}{label}_cal_cts.json', 'r') as file:
            cal_counts = json.load(file)
        print(f'Read calibrated counts from {os.getcwd()}/{label}_cal_cts.json')
    
    else:
        cirs_batch = []

        from qiskit.ignis.mitigation.measurement import complete_meas_cal
        calcirs, _ = complete_meas_cal(qubit_list=[0])
        # add two calibration circuits
        cirs_batch += calcirs

        temp = 0
        for tag in topt_dict.keys():
            # track indices in cirs_batch for each geom
            start_index = 2*nsets*temp
            end_index = start_index + 2*nsets
            topt_dict[tag]['address'] = (start_index, end_index) # store tuple
            temp += 1

            topt = topt_dict[tag]['t_opt']
            z_cir = prepare_1q_var_form('Z', topt)
            x_cir = prepare_1q_var_form('X', topt)

            for _ in range(nsets):
                cirs_batch += [z_cir, x_cir]


        n_shots = int(options['n_shots'])
        set_backend = options['backend']
        noise_model = NoiseModel.from_backend(options['device_for_noise_model']) if 'device_for_noise_model' in options.keys() else None
        print(f'\n   ==> Run a batch job <== ')    
        print(f'n_cirs            = {len(cirs_batch)} ({nsets} sets,first 2 cirs are calibration cirs)')
        print(f'n_shots per cir   = {n_shots}')
        print(f'Backend           = {label}')
        print(f'Compute RDM       = {options["compute_rdms"]}')
        # execute one circuit batch job
        job = execute(cirs_batch, shots=n_shots, backend=set_backend, noise_model=noise_model) \
            if set_backend.name() == 'qasm_simulator' else execute(cirs_batch, backend=set_backend, shots=n_shots)
        print(f'Job ID            = {job.job_id()}')

        result = job.result()
        # list of counts, the first 2 cirs are calibration circuits
        raw_counts_batch = result.get_counts()      
        with open(f'{maindir}/{l3}{label}_raw_cts.json', 'w') as file:
            json.dump(raw_counts_batch, file, indent=2)
        print(f'\nSave batch of raw counts to {os.getcwd()}/{l3}{label}_raw_cts.json')

        # do calibration using `my_oq_calibrate` function
        calcirs_counts = raw_counts_batch[:2]
        raw_counts = raw_counts_batch[2:]

        cal_counts = my_oq_calibrate(calcirs_counts, raw_counts)
        with open(f'{maindir}/{l3}{label}_cal_cts.json', 'w') as file:
            json.dump(cal_counts, file, indent=2)
        print(f'Save batch of calibrated counts to {os.getcwd()}/{l3}{label}_cal_cts.json \n')


        
    # compute energy(RDMs) from counts
    energies = {}
    for tag in topt_dict.keys():
        start_index, end_index = topt_dict[tag]['address']
        counts_ = cal_counts[start_index:end_index]
        print(f'{tag}: cir {start_index} to cir {end_index-1} in cirs_batch (removed 2 calcirs), #cirs: {len(counts_)}')
        c_z, c_x, c_0 = topt_dict[tag]['cz_cx_c0']
        es = []
        for n in range(nsets):
            counts_z = counts_[2*n]
            counts_x = counts_[2*n+1]
            (c1sq, c2sq) = get_amps_1q(counts_z)
            avg_z = c1sq - c2sq
            avg_x = get_amps_1q(counts_x)[0] - get_amps_1q(counts_x)[1]
            e = c_z*avg_z + c_x*avg_x + c_0
            es.append(e)
       
            if options['compute_rdms']:
                rdm_path = f"{maindir}/{tag}/{ints_type}/{label}_{n}"
                if not os.path.exists(rdm_path):
                    os.makedirs(rdm_path)
                os.chdir(rdm_path)
                
                rdms = compute_rdms(c1sq, c2sq, avg_x)
                rdms['energy'] = {'data': e, 'description': 'energy'}
                with open(f'rdms.json', 'w') as file:
                    json.dump(rdms, file, indent=2)
                print(f'Save rdms.json to {os.getcwd()}')
                
                os.chdir(maindir)
                           
        energies[tag] = es
        
    with open(f'{maindir}/{l3}{label.capitalize()}.json', 'w') as file:
        json.dump(energies, file, indent=2)
    print(f'\nSave energies ({nsets} sets) to {os.getcwd()}/{l3}{label.capitalize()}.json')
 
    
def save_optimal_t(options):
    maindir = options['maindir']
    ints_type = options['ints_type']
                 
    opt_t_map = {}
    for tag in options['tags']:
        opt_t_map[tag] = {}
        ints_path = f"{maindir}/{tag}/{ints_type}"
        c_z, c_x, c_0, _ = get_coeffs(ints_path, use_dressed_h=options['use_dressed_h'])
        t_A = get_optimal_t(c_x, c_z)
        opt_t_map[tag]['t_opt'] = t_A
        opt_t_map[tag]['cz_cx_c0'] = (c_z, c_x, c_0)
    
    name = f"{ints_type.lstrip('2-qldsrg').replace('/3pdc0', '_l3')}"
    with open(f"{maindir}/topt{name}.json", 'w') as file:
        json.dump(opt_t_map, file, indent=2)
    print(f"\nSave optimal t values to {maindir}/topt{name}.json")

        
if __name__ == "__main__":
    from qiskit import Aer, IBMQ
    IBMQ.load_account()
    provider = IBMQ.get_provider(project='vqe-dsrg-hardwar') # main, vqe-dsrg-hardwar

    options = {
        'backend': "ibmq_jakarta"   ,
        'n_shots': 20000  ,  # Max 100 circuits per job for ibmq_belem, 20000
        'compute_rdms': False   ,
        'use_dressed_h': True    ,
#         'device_for_noise_model': provider.get_backend('ibmq_manila') ,  
    }
    
    # change name string to backend object
    backend = Aer.get_backend('qasm_simulator') if options['backend'] == 'qasm_simulator' else provider.get_backend(options['backend'])
    options['backend'] = backend
    
    options['backend_name'] = options['backend'].name().rstrip('simulator').rstrip('_') if 'simulator' in options['backend'].name() \
    else options['backend'].name().lstrip('ibmq').lstrip('_')

    options['noise_model_name'] = '_'+options['device_for_noise_model'].name().lstrip('ibmq').lstrip('_') if 'device_for_noise_model' in options.keys() else ''


    
    ### `options['tags']` only needed by Step 1
#     options['tags'] = [
#         'bicbut',
#         'con_TS',
#         'dis_TS',
#         'g-but',
#         'gt_TS',
#         't-but',
#     ]
    options['maindir'] = "/home/renke/computations/bicbut_isomerization"
    options['ints_type'] = '2-qldsrg_casno/3pdc0'
    topt_file_map = {
        '2-qldsrg_casno'      : 'topt_casno.json'   ,
        '2-qldsrg_casno/3pdc0': 'topt_casno_l3.json',
    }
    ### Step1: save optimal t for 6 geoms to `topt_casno.json`/`topt_casno_l3.json`
#     save_optimal_t(options)
    
    ### Step2:  read `topt_casno.json` and submit a batch job containing nsets.
    with open(f"{topt_file_map[options['ints_type']]}", 'r') as file:
        topt_dict = json.load(file)
    
    nsets = 10
    run_batch_qc(options, topt_dict, nsets, restart_from_cal_counts=False)
    
    
    