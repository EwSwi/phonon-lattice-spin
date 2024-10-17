import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

#pauli matrices, i do have scalar version as well
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
identity_2 = np.eye(2, dtype=complex)

#kronecker operator for spin construction
def construct_spin_operator(n_sites, site_index, operator):
    operators = [identity_2] * n_sites
    operators[site_index] = operator
    return reduce(np.kron, operators)

#dont go over 8
n_atoms = 8

H_spin_spin = np.zeros((2**n_atoms, 2**n_atoms), dtype=complex)
H_lattice_spin = np.zeros((2**n_atoms, 2**n_atoms), dtype=complex)
H_soc = np.zeros((2**n_atoms, 2**n_atoms), dtype=complex)

#holder with no shape approx
J_values = 1.25 * (np.ones((n_atoms, n_atoms)) - np.eye(n_atoms))

for i in range(n_atoms):
    for j in range(i + 1, n_atoms):
        J_ij = J_values[i, j]
        if (i + j) % 2 == 0:
            coefficient = J_ij * 1 
        else:
            coefficient = J_ij * 1  #left from scalar singlet-triplet, i would basically 
            #use here the 1/4 aand -3/4 values in scalar approach and artificially impose AFM ordering. hence qm attempt. 
        for alpha in ['x', 'y', 'z']:
            if alpha == 'x':
                sigma = sigma_x
            elif alpha == 'y':
                sigma = sigma_y
            else:
                sigma = sigma_z
            S_i_alpha = construct_spin_operator(n_atoms, i, sigma / 2)
            S_j_alpha = construct_spin_operator(n_atoms, j, sigma / 2)
            H_spin_spin += coefficient * np.dot(S_i_alpha, S_j_alpha)

# l-s
g = 0.0
eta_lattice = 4.0
a = 1.5e-10
k_wave = np.pi / n_atoms
t = 1
u_i_t = np.cos(k_wave * np.arange(n_atoms) * a)
u_0_t = 0.1

for i in range(n_atoms):
    S_i_z = construct_spin_operator(n_atoms, i, sigma_z / 2)
    H_lattice_spin += g * u_i_t[i] * S_i_z

#soc
A_soc = -1.0

for i in range(n_atoms):
    S_i_z = construct_spin_operator(n_atoms, i, sigma_z / 2)
    H_soc += A_soc * S_i_z

H_total = H_spin_spin + H_lattice_spin + H_soc

#omega green's
energies = np.linspace(-5, 5, 500)
eta = 0.01

#diag for eigenvector
eigenvalues, eigenvectors = np.linalg.eigh(H_total)

P_up = []
P_down = []
#here im just experimenting with what i can use, basically im using projection to display up down spin matrices via green's,
# i am also trying to do some predictions based on eigenvectors, but i wouldnt rely much on these yet. 

for i in range(n_atoms):
    P_i_up = construct_spin_operator(n_atoms, i, np.array([[1, 0], [0, 0]], dtype=complex))
    P_i_down = construct_spin_operator(n_atoms, i, np.array([[0, 0], [0, 1]], dtype=complex))
    P_up.append(P_i_up)
    P_down.append(P_i_down)

#traces storage
G_trace_imag_up = []
G_trace_imag_down = []

for E in energies:
    G_E = np.linalg.inv((E + 1j * eta) * np.eye(2**n_atoms) - H_total)
    
    #im projection up down (likely not correct as of yet)
    G_E_up = sum([P @ G_E @ P for P in P_up])
    G_E_down = sum([P @ G_E @ P for P in P_down])
    
    G_trace_imag_up.append(-np.imag(np.trace(G_E_up)))
    G_trace_imag_down.append(-np.imag(np.trace(G_E_down)))
    
    
#gs found
ground_state_vector = eigenvectors[:, 0]

#eigenvector tidying 
absolute_values = np.abs(ground_state_vector)

#extracting biggest eigenvector contributions
largest_indices = np.argsort(-absolute_values)[:10]

largest_components = ground_state_vector[largest_indices]
largest_with_indices = list(zip(largest_indices, largest_components))

def index_to_spin_configuration(index, n_atoms):
    binary_str = format(index, f'0{n_atoms}b')
    #i am not sure if it is doing it right 
    spin_config = ''.join(['↑' if bit == '1' else '↓' for bit in binary_str])
    return spin_config
print("  ")

for idx, component in largest_with_indices:
    spin_config = index_to_spin_configuration(idx, n_atoms)
    print(f"lp: {idx}, ev value: {component}, spin order visual: {spin_config}")

print("gs:", eigenvalues)
ground_state_vector = eigenvectors[:, 0]
print("ev:", ground_state_vector)
plt.figure(figsize=(10, 6))
plt.plot(energies, G_trace_imag_up, label=' gp1', color='blue')
plt.plot(energies, G_trace_imag_down, label='gp2', color='red')
plt.xlabel('e', fontsize=14)
plt.ylabel('img', fontsize=14)
plt.title('', fontsize=16)
plt.grid(True)
plt.legend(fontsize=12)
plt.show()
