# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 17:27:14 2019

@author: gawel
"""
#%%
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute
from qiskit.aqua.algorithms import QuantumAlgorithm
from qiskit.aqua import AquaError, Pluggable, PluggableType, get_pluggable_class
from qiskit.aqua.algorithms.single_sample.hhl.hhl import HHL
from qiskit import Aer
from qiskit.aqua import QuantumInstance
import numpy as np
from qiskit.quantum_info import state_fidelity
from qiskit.aqua.utils.random_matrix_generator import random_hermitian
from qiskit.aqua.utils.run_circuits import find_regs_by_name
from qiskit.aqua.input import LinearSystemInput
from qiskit.converters import circuit_to_dag
from qiskit.validation.base import Obj
from qiskit.extensions.simulator import snapshot
from qiskit.compiler import transpile
from qiskit.aqua.utils import compile_circuits
from QGP_functions import *
import pickle
import copy
#from QGP_functions import vec_init_spos, entangle_hhl, plot_eigs, slice_circuit
#from gfunc.vec_initialize import vec_init_spos
#from gfunc.entngl import entngl
import time
import pprint

from qiskit.aqua.utils.controlled_circuit import get_controlled_circuit

#%%



class QGP(QuantumAlgorithm):
    
    def __init__(
        self,
        matrix=None,
        u=None,
        v=None,
        params=None
    ):
        
        self._matrix = matrix
        self._u = u
        self._v = v
        self._params = params
        self._circuit = None
        self._f1 = None
        self._ret = {}
        self._c_u = None
        self._c_iv = None
        self._stv = None
        self._snap_enabled = False
        self._f1 = None
    @classmethod
    def init_params(cls, params, u, v, matrix):
        """Initialize via parameters dictionary and algorithm input instance

        Args:
            params: parameters dictionary
            
        """        
        if not isinstance(matrix, np.ndarray):
            matrix = np.asarray(matrix)

        if not isinstance(u, np.ndarray):
            u = np.asarray(u)
        u = u.reshape(-1,1)
        if not isinstance(v, np.ndarray):
            v = np.asarray(v)
        v = v.reshape(-1,1)
        if min(v)<0.:
            raise ValueError("To ensure correct sign of the result all the entries "
                             "should be positive. Consider translating the " 
                             "problem by adding const.")

        if matrix.shape[0] != len(u):
            raise ValueError("Input vector dimension does not match input "
                             "matrix dimension!")
        
        if matrix.shape[0] != len(v):
            raise ValueError("Input vector dimension does not match input "
                             "matrix dimension!")
        if np.log2(matrix.shape[0]) % 1 != 0:
            # TODO: extend vector and matrix for nonhermitian/non 2**n size
            #  matrices and prune dimensions of HHL solution
            raise ValueError("Matrix dimension must be 2**n!")


        return cls(matrix, u, v, params)
    
    def construct_circuit(self, measurement=False, snap = False, circ_opt = False, 
                          unitary_division = False, use_cached_circuit = True):
        
        testing_mode = True
        self._circuit_stats = {}
        self._c_u = np.linalg.norm(self._u)
        u = self._u/self._c_u        
        invec = self._v 
        self._c_iv = np.linalg.norm(invec)
        invec = invec/self._c_iv
        
        algo_input = LinearSystemInput()   
        algo_input.matrix = self._matrix
        algo_input.vector = invec       
        hhl = HHL.init_params(self._params, algo_input)
        

        q = QuantumRegister(hhl._num_q, name="io")
        sanc = QuantumRegister(1, 'sanc')
        
        if testing_mode ==True:
            t0 = time.time()

        # EigenvalueEstimation (QPE)
        QC = QuantumCircuit(q)
        QC.add_register(sanc)          
        QC += hhl._eigs.construct_circuit("circuit", q)
        print('HHL QPE: ', QC.depth())
        self._circuit_stats['QPE'] = QC.depth()
        a = hhl._eigs._output_register
        a.name = 'k'   

        if unitary_division:
            s = QuantumRegister(1, 'anc')
            QC.add_register(s)
            QC.x(s)
            #QC.add_register(a)

            hhl._reciprocal._anc = s 
            umat = division_unitary(a.size)
            qc_rec = circuit_from_unitary(umat, [s, a])#[ a, s])
            QC += qc_rec 
            #hhl._reciprocal._scale = 1./(2.**(len(a)-1)-1.) #2*np.pi/hhl._eigs._evo_time/(2**(len(a)-1)-1)   #1/(2**(a.size)-1)
            #28.06
            #hhl._reciprocal._scale  = (1-2**(-len(a)))/(2**len(a)) #2**(-len(a))
            hhl._reciprocal._scale = 2**(-len(a))
            QC.x(s)
            print('Unitaryy division reciprocal')
            
        else:    
            # Reciprocal calculation with rotation
            QC += hhl._reciprocal.construct_circuit("circuit", a)
            print('Lookup reciprocal')
            s = hhl._reciprocal._anc 




        #self._recirc = hhl._reciprocal.construct_circuit("circuit", a)
        print('HHL REci: ', QC.depth())
        self._circuit_stats['Reci'] = QC.depth() - self._circuit_stats['QPE']

 
        #       
        
        # Inverse EigenvalueEstimation
        QC += hhl._eigs.construct_inverse("circuit", hhl._eigs._circuit)
                                           #hhl._eigs._input_register,
                                           #hhl._eigs._output_register)
        print('HHL QPE inverse: ', QC.depth())

        hhl._io_register = q
        hhl._eigenvalue_register = a
        hhl._ancilla_register = s
        
        qc = QuantumCircuit()
        dot = QuantumRegister(1, 'dot')
        if testing_mode ==True:
            print('Assembled HHL part, time:', time.time() - t0)
        
        #ADDING REGISTERS IN The RIGTH ORDER
        for qreg in QC.qregs: 
            if qreg.name == 'anc':
                qc.add_register(dot)
            qc.add_register(qreg)
        
        #initialization of u and v conditioned on sanc        
        qc += vec_init_spos(q, sanc, u, invec)        
        qc.x(sanc[0])
        qc.cx(sanc[0], hhl._ancilla_register[0]) #cx of D conditioned on A being  |0>
        qc.x(sanc[0])
        print(QC.depth())
        self._circuit_stats['init'] = qc.depth()
        self._circuit_stats['pure_HHL'] = QC.depth()
        if circ_opt:
            print(QC.depth())
            QC = transpile(QC, backend=Aer.get_backend('qasm_simulator'), seed_transpiler=11, optimization_level=2)
            
            print('Optimized HHL part, time:', time.time() - t0)
            print('Optimized depth: ', QC.depth())



        print(QC.count_ops())
        #controlled HHL
        #qc += QuantumCircuit().from_qasm_str(entangle_hhl(QC))
        QC_entgl = get_controlled_circuit(QC, sanc[0], use_basis_gates=False)


        qc += QC_entgl          #get_controlled_circuit(QC, sanc[0], use_basis_gates=False)  
        if testing_mode ==True:      
            print('Entagnled HHL, time:', time.time() - t0)
        #measuring observable M
        qc.h(dot[0])
        qc.cx(dot[0], sanc[0])
        qc.h(dot[0])
        qc.h(sanc[0])
        print('Entangled HHL depth: ', qc.depth())
        self._circuit_stats['entangled_HHL'] = qc.depth()

        #if True:
            #qc.x(a[1])
        
        # Measurement of the ancilla qubit and dot qubit

        if  self._snap_enabled or snap:
            qc.snapshot('1')    

        if measurement:
            qc.barrier()
            cd = ClassicalRegister(1)
            cm = ClassicalRegister(1)
            qc.add_register(cd)
            qc.add_register(cm)
            
            qc.measure(s, cd)
            qc.measure(dot, cm)
            
        self._circuit = qc
        if testing_mode ==True:
            print('Circuit compiled, time: ', time.time() - t0)
        
        self._f1 = 2. * np.pi *hhl._reciprocal._scale /hhl._eigs._evo_time
        #self._f1 = (2*np.pi*(1-2**(-len(a))))/((2**len(a)-1)*hhl._eigs._evo_time)
        return qc

    def construct_entangled_HHL_qobj(self, backend, qobj_pickle, measurement=False, snap = False, circ_opt = False, 
                          unitary_division = False):
        '''
        This is an alternative for 'construct circuit', designed to cut simulation time by reusing
        the compiled circuit (qobj), such that only the initialization routine can be repalced for given matrix
        '''

        testing_mode = True
        self._circuit_stats = {}
        self._c_u = np.linalg.norm(self._u)
        u = self._u/self._c_u        
        invec = self._v 
        self._c_iv = np.linalg.norm(invec)
        invec = invec/self._c_iv
        
        algo_input = LinearSystemInput()   
        algo_input.matrix = self._matrix
        algo_input.vector = invec       
        hhl = HHL.init_params(self._params, algo_input)
        

        q = QuantumRegister(hhl._num_q, name="io")
        sanc = QuantumRegister(1, 'sanc')
        
        if testing_mode ==True:
            t0 = time.time()

        # EigenvalueEstimation (QPE)
        QC = QuantumCircuit(q)
        QC.add_register(sanc)          
        QC += hhl._eigs.construct_circuit("circuit", q)
        print('HHL QPE: ', QC.depth())
        self._circuit_stats['QPE'] = QC.depth()
        a = hhl._eigs._output_register
        a.name = 'k'   

        if unitary_division:
            s = QuantumRegister(1, 'anc')
            QC.add_register(s)
            QC.x(s)
            hhl._reciprocal._anc = s 
            umat = division_unitary(a.size)
            qc_rec = circuit_from_unitary(umat, [s, a])#[ a, s])
            QC += qc_rec 

            hhl._reciprocal._scale = 2**(-len(a))
            QC.x(s)
            print('Unitaryy division reciprocal')
            
        else:    
            # Reciprocal calculation with rotation
            QC += hhl._reciprocal.construct_circuit("circuit", a)
            print('Lookup reciprocal')
            s = hhl._reciprocal._anc 

        #self._recirc = hhl._reciprocal.construct_circuit("circuit", a)
        print('HHL REci: ', QC.depth())
        self._circuit_stats['Reci'] = QC.depth() - self._circuit_stats['QPE']

        # Inverse EigenvalueEstimation
        QC += hhl._eigs.construct_inverse("circuit", hhl._eigs._circuit)
                                           #hhl._eigs._input_register,
                                           #hhl._eigs._output_register)
        print('HHL QPE inverse: ', QC.depth())

        hhl._io_register = q
        hhl._eigenvalue_register = a
        hhl._ancilla_register = s
        
        qc = QuantumCircuit()
        dot = QuantumRegister(1, 'dot')
        if testing_mode ==True:
            print('Assembled HHL part, time:', time.time() - t0)
        
        #ADDING REGISTERS IN The RIGTH ORDER
        for qreg in QC.qregs: 
            if qreg.name == 'anc':
                qc.add_register(dot)
            qc.add_register(qreg)
        
        #initialization of u and v conditioned on sanc        
        
        #those thow go to init qobj
        #qc += vec_init_spos(q, sanc, u, invec)        
        #qc.x(sanc[0])
        
        qc.cx(sanc[0], hhl._ancilla_register[0]) #cx of D conditioned on A being  |0>
        qc.x(sanc[0])
        print(QC.depth())
        self._circuit_stats['pure_HHL'] = QC.depth()
        if circ_opt:
            print(QC.depth())
            QC = transpile(QC, backend=Aer.get_backend('qasm_simulator'), seed_transpiler=11, optimization_level=2)
            
            print('Optimized HHL part, time:', time.time() - t0)
            print('Optimized depth: ', QC.depth())

        print(QC.count_ops())
        #controlled HHL
        #qc += QuantumCircuit().from_qasm_str(entangle_hhl(QC))
        QC_entgl = get_controlled_circuit(QC, sanc[0], use_basis_gates=False)


        qc += QC_entgl          #get_controlled_circuit(QC, sanc[0], use_basis_gates=False)  
        if testing_mode ==True:      
            print('Entagnled HHL, time:', time.time() - t0)
        #measuring observable M
        qc.h(dot[0])
        qc.cx(dot[0], sanc[0])
        qc.h(dot[0])
        qc.h(sanc[0])
        print('Entangled HHL depth: ', qc.depth())
        self._circuit_stats['entangled_HHL'] = qc.depth()

        
        # Measurement of the ancilla qubit and dot qubit

        if  self._snap_enabled or snap:
            qc.snapshot('1')    

        if measurement:
            qc.barrier()
            cd = ClassicalRegister(1)
            cm = ClassicalRegister(1)
            qc.add_register(cd)
            qc.add_register(cm)
            
            qc.measure(s, cd)
            qc.measure(dot, cm)
            
        self._circuit = qc
        if testing_mode ==True:
            print('Circuit compiled, time: ', time.time() - t0)
        
        self._f1 = 2. * np.pi *hhl._reciprocal._scale /hhl._eigs._evo_time

        qobj = compile_circuits(qc, backend =  backend)

        self._HHL_entgl_qobj = qobj
        #dumping into pickle
        f=open(qobj_pickle,'wb')
        pickle.dump(qobj, f)
        f.close()

        self._params['f1'] = self._f1

        return qobj
    

    def construct_vec_init_qobj(self, u, v, backend):
                #qc += vec_init_spos(q, sanc, u, invec)        
        #qc.x(sanc[0])
        sanc = QuantumRegister(1, 'sanc')
        q = QuantumRegister(int(np.log2(len(u))), name="io")

        qc = QuantumCircuit(q, sanc)
        qc += vec_init_spos(q, sanc, u/np.linalg.norm(u), v/np.linalg.norm(v))   
        qc.x(sanc[0])

        qobj = compile_circuits(qc, backend =  backend)
        return qobj

    def combine_qobjects(self, qobj_hhl, qobj_init):
        qobj_hhl = copy.deepcopy(qobj_hhl)
        qobj_hhl[0].experiments[0].instructions = qobj_init[0].experiments[0].instructions + qobj_hhl[0].experiments[0].instructions
        return qobj_hhl

    def _get_res_from_stv(self):
        n = len(self._v)

        qtr2 = int(len(self._stv)/4)
        half2 = int(len(self._stv)/2)
        
        stv0 = self._stv[half2:half2+1*n]         #sum(u+x)
        stv1 = self._stv[3*qtr2+n:3*qtr2+2*n]     #sum(u-x)
        
        P0 =stv0 * np.conj(stv0)
        P1 =stv1 * np.conj(stv1)
        print(' Cu',  self._c_u)
        print('Cv', self._c_iv)
        return (self._c_u * self._c_iv)*np.sum(P0-P1)/(self._f1) 
    
    def _statevector_simulation(self):
        print('CALLING EXECUTE')

        if self._quantum_instance.backend_name == 'statevector_simulator':
            
            res = self._quantum_instance.execute(self._circuit, backend_options = opts)
            self._stv = res.get_statevector()

        elif self._quantum_instance.backend_name == 'QX single-node simulator':
            qi_job = execute(self._circuit, backend=self._quantum_instance.backend, shots=1)
            qi_result = qi_job.result()
            vec_dict = Obj.to_dict(qi_result.data(self._circuit)['probabilities'])
            vec = np.zeros((int(2**(self._circuit.width()/2))) )    
            for key, value in vec_dict.items():
                vec[int(key, 16)] = value       #converting HEX key to decimal index     
            self._stv = np.sqrt(vec)

        print('EXECUTED')
        self._ret['qgp_result_stv'] = get_res_from_stv(self._stv) 

    def _statevector_simulation_sliced(self, sliced_length):
        
        print('Splitting the circuit')
        qc = slice_circuit(self._circuit, sliced_length)
        q1_temp = qc[0]
        print('Number of sub-circuits:\t', len(qc))
        print('CALLING EXECUTE')

        if self._quantum_instance.backend_name == 'statevector_simulator':
            opts = {'max_parallel_threads': 0, 'statevector_parallel_threshold': 8}
            res = self._quantum_instance.execute(q1_temp, backend_options = opts)
            self._stv = res.get_statevector()

            
        elif self._quantum_instance.backend_name == 'QX single-node simulator':
            qi_job = execute(q1_temp, backend=self._quantum_instance.backend, shots=1)
            qi_result = qi_job.result()
            vec_dict = Obj.to_dict(qi_result.data(q1_temp)['probabilities'])
            vec = np.zeros((int(2**(q1_temp.width()/2))) )    
            for key, value in vec_dict.items():
                vec[int(key, 16)] = value       #converting HEX key to decimal index     
            self._stv = np.sqrt(vec)
            

        for i in range(1, len(qc)):
            qc_init = QuantumCircuit()
            qc_init.qregs = qc[i].qregs
            qc_init.initialize(self._stv, qc_init.qregs)
            qc_temp = transpile(qc_init, basis_gates = ['u2', 'u3', 'cx'])
            qc_temp += qc[i]
            #qc_temp = transpile(qc_temp, basis_gates=
            # [ 'u2', 'u3', 'cx', 'cz', 'id', 'x', 'y', 'z',
            #            'h', 's', 'sdg', 't', 'tdg', 'ccx', 'swap',
            #            'multiplexer', 'snapshot', 'unitary', 'reset'])        

            if self._quantum_instance.backend_name == 'statevector_simulator':
                res = self._quantum_instance.execute(qc_temp)
                self._stv = res.get_statevector()

                
            elif self._quantum_instance.backend_name == 'QX single-node simulator':
                qi_job = execute(qc_temp, backend=self._quantum_instance.backend, shots=1)
                qi_result = qi_job.result()
                vec_dict = Obj.to_dict(qi_result.data(qc_temp)['probabilities'])
                vec = np.zeros((int(2**(qc_temp.width()/2))) )    
                for key, value in vec_dict.items():
                    vec[int(key, 16)] = value       #converting HEX key to decimal index     
                self._stv = np.sqrt(vec)
                print('STVM ', i, np.linalg.norm(self._stv))
                self._stv = self._stv/np.linalg.norm(self._stv) # needs to be renormalized because of rounding errors
        print('EXECUTED')

        self._ret['qgp_result_stv'] = self._get_res_from_stv()

    def get_res_from_counts(self):
        shots_t = 0
        P0 = 0
        P1 = 0
        for k, vals in self._counts.items():
                shots_t += vals     
                print(k, vals)
                
                if k[-1] == '1':   
                    if k[0] == '0':
                        P0 = vals                    
                    elif k[0] =='1':
                        P1 = vals 
        print('P0: ', P0)
        print('P1: ', P1)
        print('P success:', P0+P1/shots_t)
        print('Res (P0-P1)/shots:', (P0-P1)/shots_t)
        print(shots_t)               
        print(self._c_u * self._c_iv) 
        self._ret['qgp_result_shots'] = (self._c_u * self._c_iv)*np.sum(P0-P1)/(self._f1)/shots_t
    
         
    def _qasm_simulation(self):
        print('CALLING EXECUTE')
        res = self._quantum_instance.execute(self._circuit)   
        print('EXECUTED')
        #self._stv = res.get_snapshot('1')
        #self._ret['qgp_result_snap'] = get_res_from_stv()
        
        if self._snap_enabled:
            try:
                stv = np.array(res.data()['snapshots']['statevector']['1'][0])
                self._stv = stv[:,0] + 1j * stv[:,1]
                self._ret['qgp_result_stv'] = self._get_res_from_stv()
            except:
                print('No snapshot')

        self._counts = res.get_counts()    
        self.get_res_from_counts()

    
    def _run(self, testing_mode = False, circuit_slice = 0, snapshot_enabled = False, circuit_optimization = False,    
            unitary_division = False) :
        start = time.time()

        self._snap_enabled = snapshot_enabled

        if self._quantum_instance.backend_name == 'QX single-node simulator':
            self.construct_circuit(measurement=False, circ_opt= circuit_optimization, 
                                    unitary_division=unitary_division, 
                                    use_cached_circuit = use_cached_circuit)
            self._circuit = transpile(self._circuit, basis_gates=
             ['u2', 'u3', 'cx', 'cz', 'id', 'x', 'y', 'z',
                        'h', 's', 'sdg', 't', 'tdg', 'ccx', 'swap',
                        'multiplexer', 'snapshot', 'unitary', 'reset', 'initialize'])
                        
            c = ClassicalRegister(self._circuit.width())
            self._circuit.add_register(c)
            print('Circuit info:\n')
            pprint.pprint(self._ret["circuit_info"])
            #print('Circuit info:\n', circuit_to_dag(self._circuit).properties())

            
            if circuit_slice > 0:
                self._statevector_simulation_sliced(sliced_length = circuit_slice)
            else:
                self._statevector_simulation()

        elif self._quantum_instance.is_statevector:
            self.construct_circuit(measurement=False, circ_opt= circuit_optimization, 
                                    unitary_division=unitary_division, 
                                    use_cached_circuit = use_cached_circuit) 
            #print('Circuit info:\n', circuit_to_dag(self._circuit).properties())
            self._ret["circuit_info"] = circuit_to_dag(self._circuit).properties()
            print('Circuit info:\n')
            pprint.pprint(self._ret["circuit_info"])

            if circuit_slice > 0:
                self._statevector_simulation_sliced(sliced_length = circuit_slice)
            else:
                self._statevector_simulation()
            

        else:
            self.construct_circuit(measurement=True, circ_opt= circuit_optimization, 
                                    unitary_division=unitary_division)

            self._ret["circuit_info"] = circuit_to_dag(self._circuit).properties()
            print('Circuit info:\n')
            pprint.pprint(self._ret["circuit_info"])

            #print('Circuit info:\n', circuit_to_dag(self._circuit).properties())
            self._qasm_simulation()

        # Adding a bit of general result information
        self._ret['classical_result'] = np.dot(self._u.T, np.linalg.solve(self._matrix, self._v))
        self._ret['error%_stv'] =float(100.0*  np.abs((self._ret['qgp_result_stv'] - self._ret['classical_result'])/self._ret['classical_result']))
        
        if self._ret['qgp_result_shots'] is not None:
            self._ret['error%_shots'] =float(100.0*  np.abs((self._ret['qgp_result_shots'] - self._ret['classical_result'])/self._ret['classical_result']))

        self._ret["input_matrix"] = self._matrix
        self._ret["input_vector_v"] = self._v
        self._ret["input_vector_u"] = self._u
        self._ret["eigenvalues_classical"] = np.linalg.eig(self._matrix)[0]
        dag = circuit_to_dag(self._circuit)
        self._ret["circuit_width"] = dag.width()
        self._ret["circuit_depth"] = dag.depth()
        #self._ret["gate_count_total"] = self._circuit.number_atomic_gates()
        self._ret["execution_time"] = time.time() - start 
        #if testing_mode:
        self._ret['params'] = self._params
            #self._ret["circuit_info"] = circuit_to_dag(self._circuit).properties()
            
        return self._ret
    
        #PURE HHL
    def check_eigs(self, invec):
        algo_input = LinearSystemInput()   
        algo_input.matrix = self._matrix
        algo_input.vector = invec
        
        hhl_p = HHL.init_params(self._params, algo_input)
    
        q = QuantumRegister(hhl_p._num_q, name="io")
        QC = QuantumCircuit(q)
    
        # InitialState
        QC += hhl_p._init_state.construct_circuit("circuit", q)
    
        # EigenvalueEstimation (QPE)
        QC += hhl_p._eigs.construct_circuit("circuit", q)
        a = hhl_p._eigs._output_register
        
        c = ClassicalRegister(self._params['eigs']['num_ancillae'])    
        QC.add_register(c)    
        QC.measure(a, c)
            
        backend = Aer.get_backend('qasm_simulator')
        job = execute(QC, backend)
        result = job.result()   
        rd = result.get_counts(QC)

        rets = sorted([[rd[k], k, k] for k in rd])[::-1]
        print(rets)
        for d in rets:
            d[2] = sum([2**-(i+1) for i, e in enumerate(reversed(d[2])) if e == "1"])*2*np.pi/hhl_p._eigs._evo_time #this might be wrong... (only negative eigs)
        ret = {}
        ret['measurements'] = rets
        ret['evo_time'] = hhl_p._eigs._evo_time
        fig, x, y = plot_eigs(ret)        
        return fig, x, y, ret

'''
All the params parameters:
params = {
'algorithm': {
        'name': 'HHL',
        'num_ancillae': k, 
        'num_time_slices': 50,
        'expansion_mode': 'suzuki',
        'expansion_order': 2,
        'negative_evals':False,
        #'evo_time': 2*np.pi/4,
        #'use_basis_gates': False,
},
'eigs': {
       'name': 'EigsQPE',
        'num_ancillae': k,
        'num_time_slices': 50,
       'expansion_mode': 'suzuki',
        'expansion_order': 2,
        'negative_evals':True,
        #'evo_time': 2*np.pi/4,
},
"iqft": {
    "name": "STANDARD"
},
"qft": {
    "name": "STANDARD"
},
"initial_state": {
    "name": "CUSTOM",
    "state_vector": v,#[1/2**0.5,1/2**0.5]
},
'reciprocal': {
        'name': 'Lookup',
        #'pat_length': 4, # eigenvalue register
        #'subpat_length': 3,
        #'scale': 0,
        'negative_evals':False, 
        'evo_time': None, 
        'lambda_min': min(w),
},
}        
'''

#%%
