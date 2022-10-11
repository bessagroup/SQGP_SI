# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 00:17:43 2019

@author: gawel
"""
#%%
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.aqua.utils.run_circuits import find_regs_by_name
import numpy as np
import matplotlib.pyplot as plt
#from initializer_circuit import *
#from qiskit.extensions import initializer
from qiskit.aqua.components.initial_states.custom import Custom
from qiskit.compiler import transpile
from qiskit.aqua.utils.controlled_circuit import get_controlled_circuit
from qiskit.quantum_info.operators import Operator
import itertools

#%%
def entangle(qc):
    '''
    A simple function to entangle circuit with ancilla by modifying the qasm code
    Entangling is achieved by extending original cx gates to ccx  and ry gates to cu3, 
    where the additional condition is ancilla register named 'anc'
    
    INPUT:
        qc = QuantumCircuit(q, anc)
        register anc must be named 'sanc', and size 1
        
    OUTPUT:
        quantum circuit conditioned on ancilla
    '''
    qstr = qc.qasm()
    #print('qstr: ', qstr)
    lines = qstr.split(';\n')    
    for l, i in zip(lines, range(len(lines))): #Quick fix - some lines are separated by double \n
        if l[:1]=='\n':
            l = l[1:]
            lines[i] = l
            #print('detected')
    for l, i in zip(lines, range(len(lines))):            
        if l[:2]== 'ry':
            l = 'cu3' + l[2:]
            l  = l.split(')')
            l = l[0] + ',0,0) sanc[0], '  + l[1] 
            lines[i] = l
        if l[:2]== 'cx':
            l = 'ccx sanc[0],' + l[2:]
            lines[i] = l
                            
    en_qstr = ';\n'.join(lines)
    return(en_qstr)


def slice_circuit(QC, slice_length):
    slice_length = int(slice_length)
    qstr = QC.qasm()
    lines = qstr.split(';\n')    
    header = len(QC.qregs) + len(QC.cregs) + 2
    n_circuits = int(np.ceil(len(lines)/slice_length))
    gate_list = lines[header:-1]
    qc_list = []
    for i in range(n_circuits):
        if (i+1)*slice_length <= len(gate_list): 
            nth_circuit_gates = ';\n'.join(gate_list[i*slice_length:(i+1)*slice_length])
        else:
            nth_circuit_gates = ';\n'.join(gate_list[i*slice_length:])

                
        nth_circuit = ';\n'.join(lines[:header]) + ';\n' + nth_circuit_gates + ';\n'+lines[-1]
        qc_list.append(QuantumCircuit().from_qasm_str(nth_circuit))
    
    return(qc_list)

    
def vec_init_spos(q, sanc, u, v):
    '''
    Function initializing vectors u and v on reg. q 
    conditioned on ancilla reg. 'sanc' in state |0> and |1> respecitivelly
    
    Returns initialization circuit
    '''
        
    v = v.astype('complex')
    qcv0 = QuantumCircuit(q, sanc)
    #qcv.initialize_circuit_custom(v, q)
    #qcv += Custom(q.__len__(), state_vector= v).construct_circuit('circuit', q)    
    qcv0.initialize(v, q)
    qcv = transpile(qcv0, basis_gates = ['u1', 'u2', 'u3', 'cx'])




    u = u.astype('complex')
    qcu0 = QuantumCircuit(q, sanc)
    #qcu.initialize_circuit_custom(u, q)
    #qcu += Custom(q.__len__(), state_vector= u).construct_circuit('circuit', q)    
    qcu0.initialize(u, q)
    qcu = transpile(qcu0, basis_gates = ['u1', 'u2', 'u3', 'cx'])

    #initialized vectors entangled with ancilla        
    #qcv_e =  QuantumCircuit().from_qasm_str(entangle(qcv))
    #qcu_e = QuantumCircuit().from_qasm_str(entangle(qcu))
    qcv_e = get_controlled_circuit(qcv, sanc[0], use_basis_gates = False)    
    qcu_e = get_controlled_circuit(qcu, sanc[0], use_basis_gates = False)    

    #initializing global circuit
    QC = QuantumCircuit(q, sanc)
    QC.h(sanc[0])
    QC.x(sanc[0])                #ancilla in superposition
    QC += qcu_e                  #initializing V conditioned on ancilla = 0
    QC.x(sanc[0])                #initializing U conditioned on ancilla = 1
    QC += qcv_e    
    return QC

def entangle_ccx(qc, ccxg):
    sanc = find_regs_by_name(qc, 'sanc')    
    ccxg = ccxg.split(' ')
    regs = ccxg[1]
    regs = regs.split(',')
    ctrls = []
    for reg in regs:
        reg = reg.split('[')
        ctrls.append(find_regs_by_name(qc, reg[0])[int(reg[1][:-1])])
    
    trgt = ctrls[-1]
    ctr = ctrls[:-1] + [sanc[0]]
    qc_temp = QuantumCircuit()
    
    for iqreg in qc.qregs:
        qc_temp.add_register(iqreg)
    nregs = len(qc.qregs)
    
    qc_temp.mct(ctr, trgt, None, 'noancilla')
    qcccx = qc_temp.qasm().split(';\n')
    res_str = qcccx[2+nregs:-1]
    return(res_str)

def entangle_hhl(qc):
    '''
    A simple function to entangle circuit with ancilla by modifying the qasm code
    Entangling is achieved by extending original cx gates to ccx  and ry gates to cu3, 
    where the additional condition is ancilla register named 'anc'
    
    INPUT:
        qc = QuantumCircuit(q, anc)
        register anc must be named 'z', and size 1
        
    OUTPUT:
        quantum circuit conditioned on ancilla
    '''
    qstr = qc.qasm()
    lines = qstr.split(';\n')    
    en_lines = []
    for l, i in zip(lines, range(len(lines)+1)): #Quick fix - some lines are separated by double \n
        if l[:1]=='\n':
            l = l[1:]
            lines[i] = l
            #print('detected')
            
    for l, i in zip(lines, range(len(lines)+1)):
        if l[:2]== 'ry':
            l = 'cu3' + l[2:]
            l  = l.split(')')
            l = l[0] + ',0,0) sanc[0], '  + l[1] 
            en_lines.append(l)
        
        elif l[:2]== 'u1':
            l = 'cu1' + l[2:]
            l  = l.split(')')
            l = l[0] + ') sanc[0], '  + l[1] 
            en_lines.append(l) 
        
        elif l[:2]== 'u2':
            l = 'cu3(0.5*pi,' + l[3:]
            l  = l.split(')')
            l = l[0] + ') sanc[0], '  + l[1] 
            en_lines.append(l)
        
        elif l[:2]== 'u3':
            l = 'cu3' + l[2:]
            l  = l.split(')')
            l = l[0] + ') sanc[0], '  + l[1] 
            en_lines.append(l)
            
        elif l[:2]== 'x ':
            l = 'cx sanc[0],' + l[2:]
            en_lines.append(l)
            
        elif l[:2]== 'cx':
            l = 'ccx sanc[0],' + l[2:]
            en_lines.append(l)
        
        elif l[:3]== 'ccx':
            l = entangle_ccx(qc,l)
            en_lines += l    
        
        elif l[:2]== 'h ':
            l = 'ch sanc[0],' + l[2:]
            en_lines.append(l)
        
        
        else:
            en_lines.append(l)
        #if l[:7]== 'barrier':
        #   l = l + ', anc[0]'
        #   lines[i] = l
                            
    en_qstr = ';\n'.join(en_lines)
    return(en_qstr)
    
def plot_eigs(res):
    x = []
    y = []
    for c, s, l in res["measurements"]:
        x += [l]
        y += [c]    
    xtemp = np.sort(x)
    fig = plt.figure(figsize=(15, 4)) 
    n = len(s)   
    plt.bar(x, y/np.sum(y), width=0.9*max(x)/2**n )
    plt.xlabel('Eigenvalues')
    plt.ylabel('Probability')
    plt.grid(True)
    plt.show()
    return fig, x, y

def genDeletePos(startCol, pos_to_del, pos_h):
    '''    
    Based on Rebecca Roberts code :https://cklixx.people.wm.edu/mathlib.html    
    ''' 
    count = 0
    for k in range(np.size(startCol)):
        count = count + 1

    b = startCol[pos_to_del, 0]
    a = startCol[pos_h, 0]

    denom = np.sqrt(np.abs(a)**2 + np.abs(b)**2)
    if (denom == 0 ):
        a1 = 1
        b1 = 0
        a2 = 1
        b2 = 0
    else:
        a1 = np.conj(a)/denom
        b1 = np.conj(b)/denom
        a2 = a/denom
        b2 = -b/denom
    unitaryMatrix = np.zeros((count, count))
    
    #create an identity matrix
    for j in range(0, count):
        unitaryMatrix[j,j] = 1
    
    #create the two-level unitary matrix
    unitaryMatrix[pos_h, pos_h] = a1
    unitaryMatrix[pos_h, pos_to_del] = b1
    unitaryMatrix[pos_to_del, pos_h] = b2
    unitaryMatrix[pos_to_del, pos_to_del] = a2

    finalCol = unitaryMatrix * startCol
    return [finalCol, unitaryMatrix] 

def finalMatrix(matrixGiven, permutation):
    '''    
    Based on Rebecca Roberts code :https://cklixx.people.wm.edu/mathlib.html    
    '''
    d = len(matrixGiven)

    N = int(d*(d-1)/2.)

    finalMatrix = [None]*int(N)
    
    j = d
    count = d-1
    while count > 0:
        col = np.zeros((j, 1))
        for i in range(0, j):
            col[i,0] = matrixGiven[i, permutation[j-count-1]]
        for idx in range(0, count):
            [col_new, unitary] = genDeletePos(col, permutation[j-idx-1], permutation[j-idx-2])
            matrixGiven = unitary @ matrixGiven
            finalMatrix[N-1] = unitary
            N = N - 1
        count = count - 1
    return finalMatrix





def decompose_unitary(unitary):
    U = unitary
    d = len(U)
    #P = list(itertools.permutations(np.arange(0,d )))

    #succesful_perm = False
    i = 0
    #while i<len(P) and (succesful_perm ==False):
    '''
    for perm in itertools.permutations(np.arange(0,d )):
        A = []
        perm = P[i]
        A.append(finalMatrix(U,perm)) # The output is trasposed with t so A is a list of U.T
        A = A[0]
        Utemp = A[0].T
        
        for v in range(1, len(A)):
            Utemp = Utemp @ A[v].T 
        if np.allclose(Utemp, U):
            #succesful_perm = True
            break
        i = i+1
    '''
    #i = i-1
    
    A = []
    perm = []
    for p in range(int(d/2)):
        perm.append(int(p))
        perm.append(int(p+d/2))
    #perm = P[i]
    perm = perm[::-1]
    print('Permutation ', perm)
    A.append(finalMatrix(U,perm))
    A = A[0]

    Atr = []
    for matrix in A:
        if not np.allclose(matrix, np.eye(d)):
            Atr.append(matrix)
    
    #define dictionary for storing data on partial unitaries
    Useq = {}
    for param in ['U', 'ureg', 'cnot', 'ustat']:
        Useq[param] = []

    for i in range(len(Atr)):
        idx = np.where(Atr[i] != np.eye(d))         #delete all identity unitaries
        if len(idx[0])==2:
            U = np.diag(Atr[i][idx[0], idx[1]])
        else: 
            U = Atr[i][idx[0], idx[1]].reshape(2, 2) 

        Useq['U'].append(U)                         #append the remaining to the ditionary         
        cnot = []
        b = idx[0][-1]
        a = idx[0][0]

        bin_list = [2]    #initialize bin list as 2 to ensure firs run of the "while" loop              

        while sum(bin_list)>1:

            bin_diff = bin(b-a)[2:]
            #print(bin_diff)
            bin_list = []
            for i in range(len(bin_diff)):
                bin_list.append(int(bin_diff[i]))

            if sum(bin_list)>1:
                #print(len(bin_list))
                cnot.append([len(bin_list)-1, bin(a) ])
                a = a+2**(len(bin_list)-1)

        bin_diff = bin(b-a)[2:]
        bin_list = []
        for i in range(len(bin_diff)):
            bin_list.append(int(bin_diff[i]))

        Useq['cnot'].append(cnot)    
        Useq['ustat'].append([bin(a), bin(b)])
        Useq['ureg'].append(np.where(np.array(bin_list[::-1]) ==1))
    return Useq

#%%
def circuit_from_2level_unitaries(Useq, qreg_list, umat):
    #q = QuantumRegister(k)
    qc = QuantumCircuit()
    q  =[]
    for qreg in qreg_list:
        qc.add_register(qreg)
        j = len(qreg)
        for i in qreg[::-1]:
            #if not(qreg.name == 'k' and i[1] == 0):
            q.append(i)
            print(i)
    q = q[::-1]
    k = len(q)
    
    print('K unitary: ', k)
    print(Useq)
    for i in range(len(Useq['ustat'])):        
        for idx, bit in enumerate(Useq['ustat'][i][-1][2:][::-1]):
            if bit == '0':
                qc.x(q[idx])
                #print(i, idx)
                
        qc_u = QuantumCircuit()
        for qreg in qreg_list:
            qc_u.add_register(qreg)

        qc_u.u3( -2* np.arccos(Useq['U'][i][0][0]), 0, 0, q[int(Useq['ureg'][i][0][0])]) # negative sign to implement U.T
        
        for idx in range(len(Useq['ustat'][i][-1][2:])):
            if idx != Useq['ureg'][i][0][0]:
                qc_u = get_controlled_circuit(qc_u, q[idx])# q[1]])
        qc += qc_u       
        for idx, bit in enumerate(Useq['ustat'][i][-1][2:][::-1]):
            if bit == '0':
                qc.x(q[idx])
    np.set_printoptions(precision=3, suppress=True)
    print(umat)
    if qc.size() ==0:
        qc.append(Operator(umat), [q[0], q[1]])
    print('qc size after unitaries: ', qc.size())

    return qc            
    #res = execute(qc, backend = Aer.get_backend('unitary_simulator')).result().get_unitary()

def circuit_from_unitary(unitary, qreg_list):
    Useq = decompose_unitary(unitary)
    qc = circuit_from_2level_unitaries( Useq, qreg_list, unitary)
    return qc


def division_unitary(k):
    umat = np.zeros(((int(2**(k+1)),int(2**(k+1)))))
    for i in range(1, 2**k):
        umat[i, i] = np.sqrt(1 - (1/(i)**2))
        umat[i+2**k, i + 2**k] = np.sqrt(1 - (1/(i)**2))
        umat[i, int(i+2**k)] = 1/i
        umat[int(i+2**k), i] = -1/i
    
    # = k-1 # skipping first bit orresponding to 0 and 1
    #umat = np.zeros(((int(2**(k+1)),int(2**(k+1)))))
    #for i in range(2**k-1, 0, -1):#range(0, 2**k):
    #    umat[i, i] = np.sqrt(1 - (1/(i+1)))
    #    umat[i+2**(k), i + 2**(k)] = np.sqrt(1 - (1/(i+1)))
    #    umat[i, int(i+2**(k))] =  1/np.sqrt(i+1)
    #    umat[int(i+2**(k)), i] = -1/np.sqrt(i+1)# -np.sqrt(1 - (1/(i+1)))
    #umat[0, 0] = 1
    #umat[ int(2**k), int(2**k)] = 1
    
    return umat

#%%
