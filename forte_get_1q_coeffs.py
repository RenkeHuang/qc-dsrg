#!/home/renke/.conda/envs/psi4-2021/bin/python
import numpy as np

import psi4
import forte
from forte import forte_options

# psi4.set_memory('1 GB')  # CBD: 100 GB for QZ CBD

####################################################
# cyclobutadiene D4h geometry taken from "~/paper-collaborative/qc-dsrg/molecules/df-sq-ldsrg2-nivo_vqz/d4h/input.dat"
cbd_geom = psi4.geometry("""
C    0.7204500    0.7204500    0.0000000
C    0.7204500   -0.7204500    0.0000000
C   -0.7204500    0.7204500    0.0000000
C   -0.7204500   -0.7204500    0.0000000
H    1.4800241    1.4800241    0.0000000
H    1.4800241   -1.4800241    0.0000000
H   -1.4800241    1.4800241    0.0000000
H   -1.4800241   -1.4800241    0.0000000
noreorient
""")
cbd_mo_spaces = {
    'FROZEN_DOCC':     [0, 0, 0, 0, 0, 0, 0, 0],
    'RESTRICTED_DOCC': [4, 2, 0, 0, 0, 0, 3, 3],
    'ACTIVE':          [0, 0, 1, 1, 1, 1, 0, 0],
}
cbd_state_0 = {
    forte.Determinant([0, 1, 0, 1], [0, 1, 0, 1]): 1 / np.sqrt(2),
    forte.Determinant([1, 0, 0, 1], [1, 0, 0, 1]): -1 / np.sqrt(2),
}
cbd_state_1 = {
    forte.Determinant([1, 1, 0, 0], [0, 0, 1, 1]): 1 / np.sqrt(2),
    forte.Determinant([0, 0, 1, 1], [1, 1, 0, 0]): 1 / np.sqrt(2),
}
####################################################
h4_geom = psi4.geometry("""
H 
H 1 r 
H 2 r 1 90.0 
H 1 r 2 90.0 3 0.0 
r = 1.23 
""")
h4_mo_spaces = {
    'RESTRICTED_DOCC': [0, 0, 0, 0, 0, 0, 0, 0],
    'ACTIVE':          [1, 0, 0, 1, 0, 1, 1, 0],
}
# H4: 2 CSFs using CASSCF(4,4) orbitals for one-qubit Mapping
h4_state_0 = {
    forte.Determinant([1,0,0,1],[1,0,0,1]): 1/np.sqrt(2), 
    forte.Determinant([1,0,1,0],[1,0,1,0]): -1/np.sqrt(2)
}
h4_state_1 = {
    forte.Determinant([0,0,1,1],[1,1,0,0]):  1/np.sqrt(3),  # bbaa/--++
    forte.Determinant([1,1,0,0],[0,0,1,1]):  1/np.sqrt(3),  # aabb/++--
    forte.Determinant([0,1,0,1],[1,0,1,0]):  np.sqrt(3)/6,  # baba/-+-+
    forte.Determinant([1,0,1,0],[0,1,0,1]):  np.sqrt(3)/6,  # abab/+-+-
    forte.Determinant([1,0,0,1],[0,1,1,0]): -np.sqrt(3)/6,  # abba/+--+
    forte.Determinant([0,1,1,0],[1,0,0,1]): -np.sqrt(3)/6,  # baab/-++-
####################################################
h2_geom = psi4.geometry("""
0 1
H
H 1 1.205
""")
h2_mo_spaces = {
    'RESTRICTED_DOCC': [0, 0, 0, 0, 0, 0, 0, 0],
    'ACTIVE':          [1, 0, 0, 0, 0, 1, 0, 0],
}
# H2: 2 singlet determinants for one-qubit mapping
h2_state_0 = {forte.Determinant([1, 0], [1, 0]): 1.,  }
h2_state_1 = {forte.Determinant([0, 1], [0, 1]): 1.}
####################################################

mol_lists = {
    'cyclobutadiene': {
        'geom': cbd_geom,
        'two_states': (cbd_state_0, cbd_state_1),
        'mo_spaces': cbd_mo_spaces,
    },
    'h4':{
        'geom': h4_geom,
        'two_states': (h4_state_0, h4_state_1),
        'mo_spaces': h4_mo_spaces,
    }
    'h2': {
        'geom': h2_geom,
        'two_states': (h2_state_0, h2_state_1),
        'mo_spaces': h2_mo_spaces
    }
}

mol_name = 'h2'
mol = mol_lists[mol_name]['geom']
(state_0, state_1) = mol_lists[mol_name]['two_states']
mo_spaces = mol_lists[mol_name]['mo_spaces']

psi4_options = {
    'basis': 'cc-pV5Z',
    'reference': 'rhf',
    'scf_type': 'pk',
    'e_convergence': 1.0e-12,
    # 'd_convergence': 1.0e-10,
    # 'maxiter': 300,
}

psi4.core.set_output_file(f'{mol_name}-output.dat', False)

psi4.set_options(psi4_options)

E_scf, wfn = psi4.energy('scf', molecule=mol, return_wfn=True)

psi4_options = psi4.core.get_options()  # options = psi4 option object
psi4_options.set_current_module('FORTE')  # read options labeled 'FORTE'
forte_options.get_options_from_psi4(psi4_options)


nmopi = wfn.nmopi()
point_group = wfn.molecule().point_group().symbol()
print(f'\ndoccpi: {wfn.doccpi().to_tuple()}, point_group: {point_group}')

mo_space_info = forte.make_mo_space_info_from_map(nmopi, point_group, mo_spaces, [])
ints = forte.make_ints_from_psi4(wfn, forte_options, mo_space_info)

# the space that defines the active orbitals. We select only the 'ACTIVE' part
active_space = 'ACTIVE'
# the space(s) with non-active doubly occupied orbitals
core_spaces = ['RESTRICTED_DOCC']
as_ints = forte.make_active_space_ints(mo_space_info, ints, active_space, core_spaces)
print(f'\nNumber of active orbitals: {as_ints.nmo()}')
print(f'Symmetry of the active MOs: {as_ints.mo_symmetry()}')  # only available on 'qc' branch 


def slater(d1, d2):
    return as_ints.slater_rules(d1, d2) + as_ints.frozen_core_energy() + \
           as_ints.scalar_energy() + as_ints.nuclear_repulsion_energy()


def compute_hij(state_i, state_j):
    hij = 0
    for det1 in state_i:
        for det2 in state_j:
            hij += state_i[det1] * state_j[det2] * slater(det1, det2)
    return hij

h00 = compute_hij(state_0, state_0)
h11 = compute_hij(state_1, state_1)
h10 = compute_hij(state_1, state_0)
h01 = compute_hij(state_0, state_1)

c0 = (h00 + h11) / 2
cz = (h00 - h11) / 2
cx = h10


print(f'\n\n   ==> Info for {mol_name} One-Qubit Mapping <== ')
print(f'\nh00 = {h00}\nh11 = {h11}\nh10 = {h10}\nh01 = {h01}\n')
print(f'\nc0 = {c0}\nc_z = {cz}\ncx = {h10}\n')

#  docc                [4,2,1,0,0,1,3,3]
#  restricted_docc     [4,2,0,0,0,0,3,3]
#  active              [0,0,1,1,1,1,0,0]
#  mcscf_type          df
#  mcscf_r_convergence 6
#  mcscf_e_convergence 10
#  mcscf_maxiter       500
#  mcscf_diis_start    25
