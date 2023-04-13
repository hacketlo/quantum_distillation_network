import numpy as np
import netsquid as ns
import pydynaa as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas


from netsquid.components import ClassicalChannel, QuantumChannel
from netsquid.util.simtools import sim_time
from netsquid.util.datacollector import DataCollector
from netsquid.qubits.ketutil import outerprod
from netsquid.qubits.ketstates import s0, s1
from netsquid.qubits import operators as ops
from netsquid.qubits import qubitapi as qapi
from netsquid.protocols.nodeprotocols import NodeProtocol, LocalProtocol
from netsquid.protocols.protocol import Signals
from netsquid.nodes.node import Node
from netsquid.nodes.network import Network
# from netsquid.examples.entanglenodes import EntangleNodes
from netsquid.components.instructions import INSTR_MEASURE, INSTR_CNOT, IGate
from netsquid.components.component import Message, Port
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.components.qprocessor import QuantumProcessor
from netsquid.components.qprogram import QuantumProgram
from netsquid.qubits import ketstates as ks
from netsquid.qubits.state_sampler import StateSampler
from netsquid.components.models.qerrormodels import QuantumErrorModel
from netsquid.components.models.delaymodels import FixedDelayModel, FibreDelayModel
# from netsquid.components.models import DepolarNoiseModel, T1T2NoiseModel
from netsquid.nodes.connections import DirectConnection
from pydynaa import EventExpression
from netsquid.qubits.qformalism import QFormalism, get_qstate_formalism
from netsquid.util.constrainedmap import ValueConstraint, nonnegative_constr
import random

import heapq
import numpy as np


from netsquid.protocols.nodeprotocols import NodeProtocol
from netsquid.protocols.protocol import Signals
from netsquid.components.instructions import INSTR_SWAP
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.components.qprocessor import QuantumProcessor, PhysicalInstruction
from netsquid.qubits import ketstates as ks
from netsquid.qubits.state_sampler import StateSampler
from netsquid.components.models.delaymodels import FixedDelayModel
from netsquid.components.qchannel import QuantumChannel
from netsquid.nodes.network import Network
from pydynaa import EventExpression

import copy
import matplotlib.pyplot as plt
from IPython.display import display

class EntangleNodes(NodeProtocol):
    #MODIFIED FROM NETSQUID EXAMPLES
    """Cooperate with another node to generate shared entanglement.

    Parameters
    ----------
    node : :class:`~netsquid.nodes.node.Node`
        Node to run this protocol on.
    role : "source" or "receiver"
        Whether this protocol should act as a source or a receiver. Both are needed.
    start_expression : :class:`~pydynaa.core.EventExpression` or None, optional
        Event Expression to wait for before starting entanglement round.
    input_mem_pos : int, optional
        Index of quantum memory position to expect incoming qubits on. Default is 0.
    num_pairs : int, optional
        Number of entanglement pairs to create per round. If more than one, the extra qubits
        will be stored on available memory positions.
    name : str or None, optional
        Name of protocol. If None a default name is set.

    """
    def __init__(self, node, role, start_expression=None, input_mem_pos=0, num_pairs=1, name=None, qs_name = None):
        if role.lower() not in ["source", "receiver"]:
            raise ValueError
        self._is_source = role.lower() == "source"
        name = name if name else "EntangleNode({}, {})".format(node.name, role)
        super().__init__(node=node, name=name)
        if start_expression is not None and not isinstance(start_expression, EventExpression):
            raise TypeError("Start expression should be a {}, not a {}".format(EventExpression, type(start_expression)))
        self.start_expression = start_expression
        self._num_pairs = num_pairs
        self._mem_positions = None
        self._qsource_name = qs_name
        # Claim input memory position:
        if self.node.qmemory is None:
            raise ValueError("Node {} does not have a quantum memory assigned.".format(self.node))
        self._input_mem_pos = input_mem_pos
        
        self._qmem_input_port = self.node.qmemory.ports["qin{}".format(self._input_mem_pos)]
        self.node.qmemory.mem_positions[self._input_mem_pos].in_use = True
    
    def start(self):
        self.entangled_pairs = 0  # counter
        self._mem_positions = [self._input_mem_pos]
        self.node.qmemory.mem_positions[self._input_mem_pos].in_use = True
        # Claim extra memory positions to use (if any):
        if (self._num_pairs > 1):
            self._mem_positions.append(self._input_mem_pos+1)
            self.node.qmemory.mem_positions[self._input_mem_pos+1].in_use = True

        # Call parent start method
        return super().start()

    def stop(self):
        # Unclaim used memory positions:
        if self._mem_positions:
            for i in self._mem_positions[1:]:
                self.node.qmemory.mem_positions[i].in_use = False
            self._mem_positions = None
        # Call parent stop method
        super().stop()

    def restart(self):
        self.stop()
        self.start()

    def run(self):
        while True:
            if self.start_expression is not None:
                yield self.start_expression
            elif self._is_source and self.entangled_pairs >= self._num_pairs:
                # If no start expression specified then limit generation to one round
                break
            for mem_pos in self._mem_positions[::-1]:
                # Iterate in reverse so that input_mem_pos is handled last
                if self._is_source:
                    self.node.subcomponents[self._qsource_name].trigger()
                yield self.await_port_input(self._qmem_input_port)
                if mem_pos != self._input_mem_pos:
                    self.node.qmemory.execute_instruction(
                        INSTR_SWAP, [self._input_mem_pos, mem_pos])
                    if self.node.qmemory.busy:
                        yield self.await_program(self.node.qmemory)
                self.entangled_pairs += 1
                self.send_signal(Signals.SUCCESS, mem_pos)

    @property
    def is_connected(self):
        if not super().is_connected:
            return False
        if self.node.qmemory is None:
            return False
        if self._mem_positions is None and len(self.node.qmemory.unused_positions) < self._num_pairs - 1:
            return False
        if self._mem_positions is not None and len(self._mem_positions) != self._num_pairs:
            return False

        return True
    
class EntangleBase(NodeProtocol):

    def __init__(self, node, role, start_expression=None, input_mem_pos=0, num_pairs=1, name=None, qs_name = None, qs_name1 = None):
        if role.lower() not in ["source", "receiver"]:
            raise ValueError
        self._is_source = role.lower() == "source"
        name = name if name else "EntangleNode({}, {})".format(node.name, role)
        super().__init__(node=node, name=name)
        if start_expression is not None and not isinstance(start_expression, EventExpression):
            raise TypeError("Start expression should be a {}, not a {}".format(EventExpression, type(start_expression)))
        self.start_expression = start_expression
        self._num_pairs = num_pairs
        self._mem_positions = None
        self._qsource_name = qs_name
        self._qsource_name1 = qs_name1
        # Claim input memory position:
        if self.node.qmemory is None:
            raise ValueError("Node {} does not have a quantum memory assigned.".format(self.node))
        self._input_mem_pos = input_mem_pos
        
        self._qmem_input_port0 = self.node.qmemory.ports["qin{}".format(self._input_mem_pos)]
        self._qmem_input_port1 = self.node.qmemory.ports["qin{}".format(self._input_mem_pos+1)]
        self.node.qmemory.mem_positions[self._input_mem_pos].in_use = True
    
    def start(self):
        self.entangled_pairs = 0  # counter
        self._mem_positions = [self._input_mem_pos]
        self.node.qmemory.mem_positions[self._input_mem_pos].in_use = True
        # Claim extra memory positions to use (if any):
        if (self._num_pairs > 1):
            self._mem_positions.append(self._input_mem_pos+1)
            self.node.qmemory.mem_positions[self._input_mem_pos+1].in_use = True

        # Call parent start method
        return super().start()

    def stop(self):
        # Unclaim used memory positions:
        if self._mem_positions:
            for i in self._mem_positions[1:]:
                self.node.qmemory.mem_positions[i].in_use = False
            self._mem_positions = None
        # Call parent stop method
        super().stop()

    def restart(self):
        self.stop()
        self.start()

    def run(self):
        while True:
            if self.start_expression is not None:
                yield self.start_expression
            if self._is_source:
                self.node.subcomponents[self._qsource_name].trigger()
            yield self.await_port_input(self._qmem_input_port0)
            self.entangled_pairs += 1
            self.send_signal(Signals.SUCCESS, self._input_mem_pos)
            if self._is_source:
                self.node.subcomponents[self._qsource_name1].trigger()
            yield self.await_port_input(self._qmem_input_port1)
            self.entangled_pairs += 1
            self.send_signal(Signals.FINISHED, self._input_mem_pos+1)
           

    @property
    def is_connected(self):
        if not super().is_connected:
            return False
        if self.node.qmemory is None:
            return False
        if self._mem_positions is None and len(self.node.qmemory.unused_positions) < self._num_pairs - 1:
            return False
        if self._mem_positions is not None and len(self._mem_positions) != self._num_pairs:
            return False

        return True


class CustomLossModel(QuantumErrorModel):
    """Custom error model used to show the effect of node and fibre losses

    Parameters:        p_depol_node : dB loss on entering/leaving a fibre + passing through a WSS
                       2dB
                        

    p_depol_length :    dB loss per km of fibre.
                        0.16db

    """
    def __init__(self, p_loss_node=2, p_loss_fibre=0.16):
        super().__init__()
        self.properties['p_loss_node'] = p_loss_node
        self.properties['p_loss_fibre'] = p_loss_fibre
        self.required_properties = ['length']

    def error_operation(self, qubits, **kwargs):
 
        for qubit in qubits:
            p_node = np.power(10, - self.properties['p_loss_node'] / 10)
            p_fibre = np.power(10, - kwargs['length'] * self.properties['p_loss_fibre'] / 10)
            prob = 1- p_node*p_fibre
            if (prob>1):
                prob=1
            x = np.random.rand()
            if (x < prob):
                ns.qubits.discard(qubit)

class T1T2NoiseModel(QuantumErrorModel): #FROM netsquid.components.models.qerrormodels. Included due to small change for bug fix
    """Commonly used phenomenological noise model based on T1 and T2 times.

    Parameters
    ----------
    T1 : float
        T1 time, dictating amplitude damping component.
    T2: float
        T2 time, dictating dephasing component. Note that this is what is called
        T2 Hahn, as opposed to free induction decay T2\\*

    Raises
    ------
    ValueError
        If T1 or T2 are negative, or T2 > T1 when both are greater than zero.

    Notes
    -----
        Implementation and tests imported from the EasySquid project.

    """

    def __init__(self, T1=0, T2=0, **kwargs):
        super().__init__(**kwargs)

        def t1_constraint(t1):
            if t1 < 0:
                return False
            t2 = self.properties.get('T2', 0)
            if t1 == 0 or t2 == 0:
                return True
            if t2 > t1:
                return False
            return True

        def t2_constraint(t2):
            if t2 < 0:
                return False
            t1 = self.properties.get('T1', 0)
            if t1 == 0 or t2 == 0:
                return True
            if t2 > t1:
                return False
            return True

        self.add_property('T1', T1,
                          value_type=(int, float),
                          value_constraints=ValueConstraint(t1_constraint))
        self.add_property('T2', T2,
                          value_type=(int, float),
                          value_constraints=ValueConstraint(t2_constraint))

    @property
    def T1(self):
        """ float: T1 time, dictating amplitude damping component."""
        return self._properties['T1']

    @T1.setter
    def T1(self, value):
        self._properties['T1'] = value

    @property
    def T2(self):
        """float: T2 time, dictating dephasing component. Note that this is what
        is called T2 Hahn, as opposed to free induction decay \\*T2."""
        return self._properties['T2']

    @T2.setter
    def T2(self, value):
        self._properties['T2'] = value

    def error_operation(self, qubits, delta_time=0, **kwargs):
        """Error operation to apply to qubits.

        Parameters
        ----------
        qubits : tuple of :obj:`~netsquid.qubits.qubit.Qubit`
            Qubits to apply noise to.
        delta_time : float, optional
            Time qubits have spent on component [ns].

        """
        for qubit in qubits:
            self.apply_noise(qubit, delta_time)

    def apply_noise(self, qubit, t):
        """
        Applies noise to the qubit, depending on its T1 and T2, and the elapsed access time.

        This follows a standard noise model used in experiments, consisting of T1 and T2 times.
        This model consists of applying an amplitude damping noise (depending on T1), followed by
        dephasing noise (depending on both T1 and T2). If T1 is 0, then only dephasing is applied
        depending on T2. If T2 is 0, then only damping is applied. If both T1 and T2 are set to
        zero, then no noise is applied at all.

        Parameters
        ----------
        qubit : :obj:`netsquid.qubits.Qubit`
            Qubit to apply noise to.
        t : float
            Elapsed time to apply noise for.

        """
        # Check whether the memory is empty, if so we do nothing
        if qubit is None:
            return
        if qubit.qstate is None:
            return
        # If no T1 and T2 are given, no noise is applied
        if self.T1 == 0 and self.T2 == 0:
            return
        # Get formalism used within netsquid
        formalism = get_qstate_formalism()
        # If the formalism is density matrices, we can apply amplitude
        # damping and will hence make no approximation to the noise.
        # If we are in the stabilizer or ket formalism, we will approximate
        # using Pauli twirl noise
        if formalism not in QFormalism.ensemble_formalisms:
            # If it's just dephasing noise, then we only apply that which falls
            # into all formalisms. If there is an amplitude damping component,
            # then we approximate according to (PRA, 86, 062318)
            if self.T1 == 0:
                # Apply dephasing noise only
                # Compute the dephasing parameter from T1 and T2
                dp = np.exp(-t / self.T2)
                probZ = (1 - dp) / 2
                # Apply dephasing noise using netsquid lib
                self._random_dephasing_noise(qubit, probZ)
            else:
                # Apply approximation to general noise model (se e.g. PRA, 86, 062318)
                # This approximation is obtained by twirling the model below
                # and results in a Pauli channel
                # Compute probabilities of Pauli channel
                if self.T1 > 0:
                    probX = (1 - np.exp(-t / self.T1)) / 4
                else:
                    probX = 0.25
                probY = probX
                if self.T2 > 0:
                    probZ = (1 - np.exp(-t / self.T2)) / 2 - probX
                else:
                    probZ = 0.5 - probX
                probI = 1 - probX - probZ - probY
                # Apply Pauli noise using netsquid library
                self._random_pauli_noise(qubit, probI, probX, probY, probZ)
        else:
            # Apply standard T1 and T2 decoherence model
            # This means we first apply amplitude damping, followed
            # by dephasing noise, if applicable
            if self.T1 > 0:
                # Apply amplitude damping
                # Compute amplitude damping parameter from T1
                probAD = 1 - np.exp(- t / self.T1)
                # print("probAD: {}".format(probAD))  # XXX
                # Apply amplitude damping noise using netsquid library function
                self._random_amplitude_dampen(qubit, probAD)
            if self.T2 > 0:
                # Apply dephasing noise
                # Compute the dephasing parameter from T1 and T2 (e^(-t/T2)/sqrt(1-probAD))
                if self.T1 == 0:
                    dp = np.exp(-t * (1 / self.T2))
                else:
                    dp = np.exp(-t * (1 / self.T2 - 1 / (2 * self.T1)))
                probZ = (1 - dp) / 2
                # Apply dephasing noise using netsquid lib
                self._random_dephasing_noise(qubit, probZ)

    def _random_amplitude_dampen(self, qubit, probAD, cache=True, cache_precision=-1):
        # For now just apply standard netsquid noise, no special DM probabilistic DM treatment
        ns.qubits.qubitapi.amplitude_dampen(qubit, probAD, prob=1, cache=cache, cache_precision=cache_precision)

    def _random_dephasing_noise(self, qubit, probZ):
        self._random_pauli_noise(qubit, 1 - probZ, 0, 0, probZ)

    def _random_pauli_noise(self, qubit, probI, probX, probY, probZ):
        # For now, just apply standard noise.
        ns.qubits.qubitapi.apply_pauli_noise(qubit, (probI, probX, probY, probZ))

class CustomDecoherenceModel(QuantumErrorModel):
    """Custom physical noise model used to show the effect of node and fibre decoherence

 
    """
    def __init__(self, p_depol_node=0.01, p_depol_length=0.016):
        super().__init__()
        self.properties['p_depol_length'] = p_depol_length
        self.required_properties = ['length']

    def error_operation(self, qubits, **kwargs):
 
        for qubit in qubits:
            if(qubit.qstate==None):
                return
            delta = np.power(10, (-self.properties['p_depol_length'])/ 10)
            prob = 1- np.exp(-delta*kwargs['length'])
            if (prob>1):
                prob=1
            ns.qubits.depolarize(qubit, prob=prob)

def example_network_setup(source_delay=5.5*10e3, source_fidelity_sq=0.994, 
               qprocessor_positions = 100, t1= 3600 * 1e9, t2=1.46e9, physical = True):

    network = Network("purify_network")

    _INSTR_Rx = IGate("Rx_gate", ops.create_rotation_op(np.pi / 2, (1, 0, 0)))
    _INSTR_RxC = IGate("RxC_gate", ops.create_rotation_op(np.pi / 2, (1, 0, 0), conjugate=True))

    for node in ('A', 'B', 'C', 'D', 'E', 'F'):
        n = network.add_node(f"node_{node}")
    
        if (physical == True):
            n.add_subcomponent(QuantumProcessor(
                f"QuantumMemory_{node}0", num_positions=qprocessor_positions, fallback_to_nonphysical=True, memory_noise_models=T1T2NoiseModel(t1, t2), phy_instructions =[
                PhysicalInstruction(_INSTR_Rx, duration=5, parallel=True, physical=True),
                PhysicalInstruction(_INSTR_RxC, duration=5, parallel=True, physical=True),
                PhysicalInstruction(INSTR_CNOT, duration=20, parallel=True,  physical=True),
                PhysicalInstruction(INSTR_MEASURE, duration=35, parallel=True, physical=True),
                PhysicalInstruction(INSTR_SWAP, duration=20, parallel=True,  physical=True)]   ))

        else:
            n.add_subcomponent(QuantumProcessor(
                f"QuantumMemory_A{node}0", num_positions=qprocessor_positions, fallback_to_nonphysical=True, memory_noise_models=T1T2NoiseModel(t1, t2)  ))
        

        state_sampler = StateSampler(
            [ks.b01, ks.s00],
            probabilities=[source_fidelity_sq, (1 - source_fidelity_sq)])
        
        for i in range(qprocessor_positions):
            n.add_subcomponent(QSource(
                f"QSource_{node}{i}", state_sampler=state_sampler,
                models={"emission_delay_model": FixedDelayModel(delay=source_delay)},
                num_ports=2, status=SourceStatus.EXTERNAL))
    
    
    return network


class DistilFurther(NodeProtocol):

    # set basis change operators for local DEJMPS step
    _INSTR_Rx = IGate("Rx_gate", ops.create_rotation_op(np.pi / 2, (1, 0, 0)))
    _INSTR_RxC = IGate("RxC_gate", ops.create_rotation_op(np.pi / 2, (1, 0, 0), conjugate=True))

    def __init__(self, node, port, role, start_expression=None, msg_header="distil", name=None, con_num = 0, distilled_pos = 1, imp = 0):
        if role.upper() not in ["A", "B"]:
            raise ValueError
        conj_rotation = role.upper() == "B"
        if not isinstance(port, Port):
            raise ValueError("{} is not a Port".format(port))
        name = name if name else "DistilNode({}, {}, {})".format(node.name, port.name, con_num)
        super().__init__(node, name=name)
        self.port = port
        self.instance_id = con_num
        self.start_expression = start_expression
        self._program = self._setup_dejmp_program(conj_rotation)
        self.local_qcount = 0
        self.local_meas_result = None
        self.remote_qcount = 0
        self.remote_meas_result = None
        self. input_mem_pos = imp
        self.header = f"distil_{con_num}"
        self._qmem_positions = [None, None]
        self.distilled_pos= distilled_pos
        self._waiting_on_second_qubit = False
        if start_expression is not None and not isinstance(start_expression, EventExpression):
            raise TypeError("Start expression should be a {}, not a {}".format(EventExpression, type(start_expression)))

    def _setup_dejmp_program(self, conj_rotation):
        INSTR_ROT = self._INSTR_Rx if not conj_rotation else self._INSTR_RxC
        prog = QuantumProgram(num_qubits=2)
        q1, q2 = prog.get_qubit_indices(2)
        prog.apply(INSTR_ROT, [q1], physical=True)
        prog.apply(INSTR_ROT, [q2], physical=True)
        prog.apply(INSTR_CNOT, [q1, q2], physical=True)
        prog.apply(INSTR_MEASURE, q2, output_key="m", inplace=False, physical=True)
        return prog

    def run(self):
        cchannel_ready = self.await_port_input(self.port)
        qmemory_ready = self.start_expression
        
        while True:
            expr = yield cchannel_ready | qmemory_ready 

            if expr.first_term.value:
                classical_message = self.port.rx_input()
                if classical_message.meta["header"]==self.header:
                    self.remote_qcount, self.remote_meas_result = classical_message.items
                if classical_message.meta["header"]==f"FAIL{self.header}":
                    # self._qmem_positions = [self.input_mem_pos, self.input_mem_pos+1]
                    self._clear_qmem_positions()
                    self.send_signal(Signals.FAIL, self.local_qcount)
                    self.local_meas_result = None
                    self._waiting_on_second_qubit = False
                    self.local_qcount=0
                    self.remote_meas_result = None
            elif expr.second_term.value:
                source_protocol = expr.second_term.atomic_source
                ready_signal = source_protocol.get_signal_by_event(
                    event=expr.second_term.triggered_events[0], receiver=self)
                if (self.node.qmemory.mem_positions[ready_signal.result]._qubit.qstate==None): #heralding
                    self._qmem_positions[0] = self.distilled_pos
                    self._qmem_positions[1] = self.input_mem_pos
                    self._clear_qmem_positions()
                    self.send_signal(Signals.FAIL, self.local_qcount)
                    self.failed = True
                    self.port.tx_output(Message([self.local_qcount, self.local_meas_result],
                                    header=f"FAIL{self.header}"))
                    self._waiting_on_second_qubit = False
                    self.local_meas_result = None
                    self.local_qcount=0
                    self.remote_qcount = 0
                    self.remote_meas_result = None
                else:
                    yield from self._handle_new_qubit(ready_signal.result)
            self._check_success()

    def start(self):
        # Clear qubits and initialise counters
        self._clear_qmem_positions()
        self.local_qcount = 0
        self.local_meas_result = None
        self.remote_qcount = 0
        self.remote_meas_result = None
        self._waiting_on_second_qubit = False
        return super().start()

    def _clear_qmem_positions(self):
        positions = [pos for pos in self._qmem_positions if pos is not None]
        if len(positions) > 0:
            self.node.qmemory.pop(positions=positions, skip_noise=True)
        self._qmem_positions = [None, None]

    def _handle_new_qubit(self, memory_position):
    # get indexes of new qubit and held quibit
        self._qmem_positions[0] = self.distilled_pos
        assert not self.node.qmemory.mem_positions[memory_position].is_empty
        assert not self.node.qmemory.mem_positions[self._qmem_positions[0]].is_empty
        assert memory_position != self._qmem_positions[0]
        self._qmem_positions[1] = memory_position
        self._waiting_on_second_qubit = False
        yield from self._node_do_DEJMPS()


    def _node_do_DEJMPS(self):
        # Perform DEJMPS distillation protocol locally on one node
        pos1, pos2 = self._qmem_positions
    
        #pos1, pos2 = self.node.qmemory.used_positions[0], self.node.qmemory.used_positions[1]
        if self.node.qmemory.busy:
            #yield self.await_program(self.node.qmemory)
            yield self.await_timer(np.random.randint(100,500))
        
        yield self.node.qmemory.execute_program(self._program, [pos1, pos2]) 
        self.local_meas_result = self._program.output["m"][0]
        self._qmem_positions[1] = None
        # Send local results to the remote node t
        self.port.tx_output(Message([self.local_qcount, self.local_meas_result],
                                    header=self.header))

    def _check_success(self):
        # Check if distillation succeeded by comparing local and remote results
        if (self.local_qcount == self.remote_qcount and
                self.local_meas_result is not None and
                self.remote_meas_result is not None):
            if self.local_meas_result == self.remote_meas_result:
                # SUCCESS
                self.send_signal(Signals.SUCCESS, self._qmem_positions[0])
            else:
                # FAILURE
                self._clear_qmem_positions()
                self.send_signal(Signals.FAIL, self.local_qcount)
            self.local_meas_result = None
            self.remote_meas_result = None
            self._qmem_positions = [None, None]

    @property
    def is_connected(self):
        if self.start_expression is None:
            return False
        if not self.check_assigned(self.port, Port):
            return False
        if not self.check_assigned(self.node, Node):
            return False
        if self.node.qmemory.num_positions < 2:
            return False
        return True

class Distil(NodeProtocol):

    # set basis change operators for local DEJMPS step
    _INSTR_Rx = IGate("Rx_gate", ops.create_rotation_op(np.pi / 2, (1, 0, 0)))
    _INSTR_RxC = IGate("RxC_gate", ops.create_rotation_op(np.pi / 2, (1, 0, 0), conjugate=True))
    # _INSTR_Rx_P = PhysicalInstruction(_INSTR_Rx, duration=1)
    # _INSTR_RxC_P = PhysicalInstruction(_INSTR_RxC, duration=1)
    # INSTR_CNOT_P = PhysicalInstruction(INSTR_CNOT, duration=1)
    # INSTR_MEASURE_P = PhysicalInstruction(INSTR_MEASURE, duration=1)

    def __init__(self, node, port, role, start_expression=None, start_expression_qb2=None, msg_header="distil", name=None, con_num = 0, imp = 0):
        if role.upper() not in ["A", "B"]:
            raise ValueError
        conj_rotation = role.upper() == "B"
        if not isinstance(port, Port):
            raise ValueError("{} is not a Port".format(port))
        name = name if name else "DistilNode({}, {}, {})".format(node.name, port.name, con_num)
        super().__init__(node, name=name)
        self.port = port
        self.instance_id = con_num
        self.start_expression = start_expression
        self.start_expression_qb2 = start_expression_qb2
        self._program = self._setup_dejmp_program(conj_rotation)
        self.local_qcount = 0
        self.local_meas_result = None
        self.remote_qcount = 0
        self.failed = False
        self. input_mem_pos = imp
        self.remote_meas_result = None
        self.header = f"distil_{con_num}"
        self._qmem_positions = [None, None]
        self._waiting_on_second_qubit = False
        if start_expression is not None and not isinstance(start_expression, EventExpression):
            raise TypeError("Start expression should be a {}, not a {}".format(EventExpression, type(start_expression)))

    def _setup_dejmp_program(self, conj_rotation):
        INSTR_ROT = self._INSTR_Rx if not conj_rotation else self._INSTR_RxC
 
        prog = QuantumProgram(num_qubits=2)
        q1, q2 = prog.get_qubit_indices(2)
        prog.apply(INSTR_ROT, [q1], physical = True)
        prog.apply(INSTR_ROT, [q2], physical=True)
        prog.apply(INSTR_CNOT, [q1, q2], physical=True)
        prog.apply(INSTR_MEASURE, q2, output_key="m", inplace=False, physical=True)
        return prog

    def run(self):
        cchannel_ready = self.await_port_input(self.port)
        qmemory_ready = self.start_expression
        qmemory2_ready = self.start_expression_qb2

        while True:
            # self.send_signal(Signals.WAITING)
            expr = yield cchannel_ready | (qmemory_ready | qmemory2_ready)

            if expr.first_term.value:
                classical_message = self.port.rx_input()
                if classical_message.meta["header"]==self.header:
                    self.remote_qcount, self.remote_meas_result = classical_message.items
                if classical_message.meta["header"]=="FAIL":
                    # self._qmem_positions = [self.input_mem_pos, self.input_mem_pos+1]
                    self._clear_qmem_positions()
                    self.send_signal(Signals.FAIL, self.local_qcount)
                    self.local_meas_result = None
                    self._waiting_on_second_qubit = False
                    self.local_qcount=0
                    self.remote_meas_result = None

            elif expr.second_term.first_term.value:
                source_protocol = expr.second_term.first_term.atomic_source
                ready_signal = source_protocol.get_signal_by_event(
                    event=expr.second_term.triggered_events[0], receiver=self)
                if (self.node.qmemory.mem_positions[ready_signal.result]._qubit.qstate==None): #heralding
                    self._qmem_positions[0] = self.input_mem_pos
                    if self.local_qcount==0:
                        yield qmemory2_ready
                        self._qmem_positions[1] = self.input_mem_pos+1
                    self._clear_qmem_positions()
                    self.send_signal(Signals.FAIL, self.local_qcount)
                    self.failed = True
                    self.port.tx_output(Message([self.local_qcount, self.local_meas_result],
                                    header="FAIL"))
                    self._waiting_on_second_qubit = False
                    self.local_meas_result = None
                    self.local_qcount=0
                    self.remote_qcount = 0
                    self.remote_meas_result = None
                else:
                    yield from self._handle_new_qubit(ready_signal.result)
                
            elif expr.second_term.second_term.value:
                source_protocol = expr.second_term.second_term.atomic_source
                ready_signal = source_protocol.get_signal_by_event(
                    event=expr.second_term.triggered_events[0], receiver=self)
                if (self.node.qmemory.mem_positions[ready_signal.result]._qubit.qstate==None): #heralding
                    self._qmem_positions[1] = self.input_mem_pos+1
                    self._clear_qmem_positions()
                    self.send_signal(Signals.FAIL, self.local_qcount)
                    self.failed = True
                    self.port.tx_output(Message([self.local_qcount, self.local_meas_result],
                                    header="FAIL"))
                    self._waiting_on_second_qubit = False
                    self.local_meas_result = None
                    self.local_qcount=0
                    self.remote_qcount = 0
                    self.remote_meas_result = None
                else:
                    yield from self._handle_new_qubit(ready_signal.result)
     
            self._check_success()

    def start(self):
        # Clear any held qubits
        self._clear_qmem_positions()
        self.local_qcount = 0
        self.local_meas_result = None
        self.remote_qcount = 0
        self.remote_meas_result = None
        self._waiting_on_second_qubit = False
        return super().start()

    def _clear_qmem_positions(self):
        positions = [pos for pos in self._qmem_positions if pos is not None]
        if len(positions) > 0:
            self.node.qmemory.pop(positions=positions, skip_noise=True)
        self._qmem_positions = [None, None]

    def _handle_new_qubit(self, memory_position):
    # Process signalling of new entangled qubit
        
        if self._waiting_on_second_qubit:
            # Second qubit arrived: perform distil
            assert not self.node.qmemory.mem_positions[self._qmem_positions[0]].is_empty
            assert memory_position != self._qmem_positions[0]
            self._qmem_positions[1] = memory_position
            self._waiting_on_second_qubit = False
            yield from self._node_do_DEJMPS()
        else:
           
            pop_positions = [p for p in self._qmem_positions if p is not None and p != memory_position]
            if len(pop_positions) > 0:
                self.node.qmemory.pop(positions=pop_positions)
            # Set new position:
            self._qmem_positions[0] = memory_position
            self._qmem_positions[1] = None
            self.local_qcount += 1
            self.local_meas_result = None
            self._waiting_on_second_qubit = True

    def _node_do_DEJMPS(self):
        # Perform DEJMPS distillation protocol locally on one node
        pos1, pos2 = self._qmem_positions
    
        #pos1, pos2 = self.node.qmemory.used_positions[0], self.node.qmemory.used_positions[1]
        if self.node.qmemory.busy:
            #yield self.await_program(self.node.qmemory)
            yield self.await_timer(np.random.randint(100,500))
        
        yield self.node.qmemory.execute_program(self._program, [pos2, pos1])  
        self.failed=False
        self.local_meas_result = self._program.output["m"][0]
        self._qmem_positions[0] = None
        # Send local results to the remote node 
        self.port.tx_output(Message([self.local_qcount, self.local_meas_result],
                                    header=self.header))

    def _check_success(self):
        # Check if distillation succeeded by comparing local and remote results
        if (self.local_qcount == self.remote_qcount and
                self.local_meas_result is not None and
                self.remote_meas_result is not None):
            if self.local_meas_result == self.remote_meas_result:
                # SUCCESS
                self.send_signal(Signals.SUCCESS, self._qmem_positions[1])
            else:
                # FAILURE
                self._clear_qmem_positions()
                self.send_signal(Signals.FAIL, self.local_qcount)
              
            self.local_meas_result = None
            self.remote_meas_result = None
            self.local_qcount=0
            self.remote_qcount = 0
            self._qmem_positions = [None, None]

    @property
    def is_connected(self):
        if self.start_expression is None:
            return False
        if not self.check_assigned(self.port, Port):
            return False
        if not self.check_assigned(self.node, Node):
            return False
        if self.node.qmemory.num_positions < 2:
            return False
        return True

def setup_data_collector(source_protocol, node_a, node_b,network_load=0, dist =1):
    def record_run(evexpr): 
        # Callback that collects data each run
        protocol = evexpr.triggered_events[-1].source
        result = protocol.get_signal_result(Signals.SUCCESS)
        # Record fidelity
        q_A, = node_a.qmemory.pop(positions=[result["pos_A"]])
        q_B, = node_b.qmemory.pop(positions=[result["pos_B"]])
        f2 = qapi.fidelity([q_A, q_B], ks.b01, squared=True)
        return {"F2": f2, "pairs": result["pairs"], "time": result["time"],  
              "network_load": network_load,
                "dist": dist }

    dc = DataCollector(record_run, include_time_stamp=False,
                       include_entity_name=False)
    dc.collect_on(pd.EventExpression(source=source_protocol,
                                     event_type=Signals.SUCCESS.value))

    return dc

class EntangleSetup(LocalProtocol):

    def __init__(self, node_a, node_b, inputmempos_a, inputmempos_b, num_runs, con_name, qs_no, source_delay=5.5*10e3 ):
        super().__init__(nodes={"A": node_a, "B": node_b}, name="Distillation example")
        nodeAltr= node_a.name[5]
        qs2 = str(int(qs_no)+1)
        self.num_runs = num_runs
        self.source_delay =source_delay

        # Initialise sub-protocols
        self.add_subprotocol(EntangleBase(node=node_a, role="source", input_mem_pos=inputmempos_a,num_pairs=2, 
                            name="entangle_A", qs_name= f"QSource_{nodeAltr}{qs_no}", qs_name1 = f"QSource_{nodeAltr}{qs2}"))
        self.add_subprotocol(EntangleBase(node=node_b, role="receiver", input_mem_pos=inputmempos_b, num_pairs=2,
                          name="entangle_B", qs_name = f"QSource_{nodeAltr}{qs_no}", qs_name1 = f"QSource_{nodeAltr}{qs2}"))
        
        # Set start expressions
        start_expr_ent_A = (self.subprotocols["entangle_A"].await_signal(
                                self, Signals.WAITING))
        start_expr_ent_B = (self.subprotocols["entangle_B"].await_signal(
                                self, Signals.WAITING))
        
        self.subprotocols["entangle_A"].start_expression = start_expr_ent_A
        self.subprotocols["entangle_B"].start_expression = start_expr_ent_B
        

    def run(self):
        for i in range(self.num_runs):
            self.start_subprotocols()
            start_time = sim_time()
            self.subprotocols["entangle_A"].entangled_pairs = 0
            self.send_signal(Signals.WAITING)
            yield (self.await_signal(self.subprotocols["entangle_A"], Signals.FINISHED) &
                   self.await_signal(self.subprotocols["entangle_B"], Signals.FINISHED))
            
            signal_A = self.subprotocols["entangle_A"].get_signal_result(Signals.SUCCESS, self)
            signal_B = self.subprotocols["entangle_B"].get_signal_result(Signals.SUCCESS, self)

            if (self._nodes._data['A'].qmemory.mem_positions[signal_A]._qubit.qstate==None or self._nodes._data['B'].qmemory.mem_positions[signal_B]._qubit.qstate==None):
                yield self.await_timer(duration=self.source_delay)
                self.send_signal(Signals.FAIL)

            else:
                result = {
                    "pos_A": signal_A,
                    "pos_B": signal_B,
                    "time": sim_time() - start_time,
                    "pairs": self.subprotocols["entangle_A"].entangled_pairs,
                }
                self.send_signal(Signals.SUCCESS, result)


class DistilSetup(LocalProtocol):

    def __init__(self, node_a, node_b, inputmempos_a, inputmempos_b, num_runs, con_name, qs_no, source_delay=5.5*10e3 ):
        super().__init__(nodes={"A": node_a, "B": node_b}, name="Distillation example")
        nodeAltr= node_a.name[5]
        qs2 = str(int(qs_no)+1)
        self.num_runs = num_runs
        self.source_delay =source_delay

        # Initialise sub-protocols
        self.add_subprotocol(EntangleBase(node=node_a, role="source", input_mem_pos=inputmempos_a,num_pairs=2, 
                            name="entangle_A", qs_name= f"QSource_{nodeAltr}{qs_no}", qs_name1 = f"QSource_{nodeAltr}{qs2}"))
        self.add_subprotocol(EntangleBase(node=node_b, role="receiver", input_mem_pos=inputmempos_b, num_pairs=2,
                          name="entangle_B", qs_name = f"QSource_{nodeAltr}{qs_no}", qs_name1 = f"QSource_{nodeAltr}{qs2}"))
        
        self.add_subprotocol(Distil(node_a, node_a.get_conn_port(node_b.ID, con_name), "A",
                                     name="purify_A", con_num = con_name, imp = inputmempos_b))
        self.add_subprotocol(Distil(node_b, node_b.get_conn_port(node_a.ID, con_name), "B",
                                     name="purify_B",con_num = con_name, imp = inputmempos_b))
        # Set start expressions
        self.subprotocols["purify_A"].start_expression = (
            self.subprotocols["purify_A"].await_signal(self.subprotocols["entangle_A"],
                                                       Signals.SUCCESS))
        self.subprotocols["purify_B"].start_expression = (
            self.subprotocols["purify_B"].await_signal(self.subprotocols["entangle_B"],
                                                       Signals.SUCCESS))
        
        self.subprotocols["purify_A"].start_expression_qb2 = (
            self.subprotocols["purify_A"].await_signal(self.subprotocols["entangle_A"],
                                                       Signals.FINISHED))
        self.subprotocols["purify_B"].start_expression_qb2 = (
            self.subprotocols["purify_B"].await_signal(self.subprotocols["entangle_B"],
                                                       Signals.FINISHED))
        

        start_expr_ent_A = (self.subprotocols["entangle_A"].await_signal(
                            self.subprotocols["purify_A"], Signals.FAIL) |
                            self.subprotocols["entangle_A"].await_signal(
                                self, Signals.WAITING))
        
        start_expr_ent_B = (self.subprotocols["entangle_B"].await_signal(
                            self.subprotocols["purify_B"], Signals.FAIL) |
                            self.subprotocols["entangle_B"].await_signal(
                                self, Signals.WAITING))
        
        self.subprotocols["entangle_A"].start_expression = start_expr_ent_A
        self.subprotocols["entangle_B"].start_expression = start_expr_ent_B
        

    def run(self):
        self.start_subprotocols()
        for i in range(self.num_runs):

            start_time = sim_time()
            self.subprotocols["entangle_A"].entangled_pairs = 0
            self.send_signal(Signals.WAITING)
            yield (self.await_signal(self.subprotocols["purify_A"], Signals.SUCCESS) &
                   self.await_signal(self.subprotocols["purify_B"], Signals.SUCCESS))

            signal_A = self.subprotocols["purify_A"].get_signal_result(Signals.SUCCESS,
                                                                       self)
            signal_B = self.subprotocols["purify_B"].get_signal_result(Signals.SUCCESS,
                                                                       self)
            
            if (self._nodes._data['A'].qmemory.mem_positions[signal_A]._qubit.qstate==None or self._nodes._data['B'].qmemory.mem_positions[signal_B]._qubit.qstate==None):
                # failtime = {
                #             "time": sim_time() - start_time
                #             }
                yield self.await_timer(duration=self.source_delay) #Cannot trigger Qsource yet
                self.send_signal(Signals.FAIL)
            
            else:
                result = {
                "pos_A": signal_A,
                "pos_B": signal_B,
                "time": sim_time() - start_time,
                "pairs": self.subprotocols["entangle_A"].entangled_pairs,
                }
                self.send_signal(Signals.SUCCESS, result)

    
            
class DistilTwice(LocalProtocol):

    def __init__(self, node_a, node_b, inputmempos_a, inputmempos_b, num_runs, con_name, qs_no, source_delay=5.5*10e3 ):
        super().__init__(nodes={"A": node_a, "B": node_b}, name="Distillation example")
        nodeAltr= node_a.name[5]
        qs2 = str(int(qs_no)+1)
        qs3 = str(int(qs_no)+2)
        self.num_runs = num_runs
        self.source_delay =source_delay

        # Initialise sub-protocols
        self.add_subprotocol(EntangleBase(node=node_a, role="source", input_mem_pos=inputmempos_a,num_pairs=2, 
                            name="entangle_A", qs_name= f"QSource_{nodeAltr}{qs_no}", qs_name1 = f"QSource_{nodeAltr}{qs2}"))
        self.add_subprotocol(EntangleBase(node=node_b, role="receiver", input_mem_pos=inputmempos_b, num_pairs=2,
                          name="entangle_B", qs_name = f"QSource_{nodeAltr}{qs_no}", qs_name1 = f"QSource_{nodeAltr}{qs2}"))

        self.add_subprotocol(EntangleNodes(node=node_a, role="source", input_mem_pos=2,
                                           num_pairs=1, name="entangle_A1", qs_name= f"QSource_{nodeAltr}{qs3}"))
        self.add_subprotocol(EntangleNodes(node=node_b, role="receiver", input_mem_pos=2, 
                                           num_pairs=1,name="entangle_B1", qs_name = f"QSource_{nodeAltr}{qs3}"))

        self.add_subprotocol(Distil(node_a, node_a.get_conn_port(node_b.ID, con_name), "A",
                                     name="purify_A", con_num = con_name, imp = inputmempos_a))
        self.add_subprotocol(Distil(node_b, node_b.get_conn_port(node_a.ID, con_name), "B",
                                     name="purify_B",con_num = con_name, imp = inputmempos_b))
        
        self.add_subprotocol(DistilFurther(node_a, node_a.get_conn_port(node_b.ID, "AB1"), "A",
                                     name="purify_A1", con_num = "AB1", distilled_pos = 1, imp = inputmempos_a + 2))
        self.add_subprotocol(DistilFurther(node_b, node_b.get_conn_port(node_a.ID, "AB1"), "B",
                                     name="purify_B1",con_num = "AB1", distilled_pos = 1, imp = inputmempos_b + 2))

        self.subprotocols["purify_A"].start_expression = (
            self.subprotocols["purify_A"].await_signal(self.subprotocols["entangle_A"],
                                                       Signals.SUCCESS))
        self.subprotocols["purify_B"].start_expression = (
            self.subprotocols["purify_B"].await_signal(self.subprotocols["entangle_B"],
                                                       Signals.SUCCESS))
        self.subprotocols["purify_A"].start_expression_qb2 = (
            self.subprotocols["purify_A"].await_signal(self.subprotocols["entangle_A"],
                                                       Signals.FINISHED))
        self.subprotocols["purify_B"].start_expression_qb2 = (
            self.subprotocols["purify_B"].await_signal(self.subprotocols["entangle_B"],
                                                       Signals.FINISHED))
        
        start_expr_ent_A = (self.subprotocols["entangle_A"].await_signal(
                            self.subprotocols["purify_A"], Signals.FAIL)|
                            self.subprotocols["entangle_A"].await_signal(
                            self.subprotocols["purify_A1"], Signals.FAIL)|
                            self.subprotocols["entangle_A"].await_signal(
                                self, Signals.WAITING))
        
        start_expr_ent_B = (self.subprotocols["entangle_B"].await_signal(
                            self.subprotocols["purify_B"], Signals.FAIL) |
                            self.subprotocols["entangle_B"].await_signal(
                            self.subprotocols["purify_B1"], Signals.FAIL)|
                            self.subprotocols["entangle_B"].await_signal(
                                self, Signals.WAITING))
        
        self.subprotocols["entangle_A"].start_expression = start_expr_ent_A
        self.subprotocols["entangle_B"].start_expression = start_expr_ent_B

        # Set start expressions
        
        self.subprotocols["purify_A1"].start_expression = (
            self.subprotocols["purify_A1"].await_signal(self.subprotocols["entangle_A1"],
                                                       Signals.SUCCESS))
        self.subprotocols["purify_B1"].start_expression = (
            self.subprotocols["purify_B1"].await_signal(self.subprotocols["entangle_B1"],
                                                       Signals.SUCCESS))


        start_expr_ent_A1 = (self.await_signal(
                             self.subprotocols["purify_A"], Signals.SUCCESS)& 
                             self.await_signal(
                             self.subprotocols["purify_B"], Signals.SUCCESS))
            
        self.subprotocols["entangle_A1"].start_expression = start_expr_ent_A1
        self.subprotocols["entangle_B1"].start_expression = start_expr_ent_A1


    def run(self):
        self.start_subprotocols()
        for i in range(self.num_runs):
     
            start_time = sim_time()
            self.subprotocols["entangle_A"].entangled_pairs = 0
            self.subprotocols["entangle_A1"].entangled_pairs = 0
            self.send_signal(Signals.WAITING)

            yield ( (self.await_signal(self.subprotocols["purify_A1"], Signals.SUCCESS) & self.await_signal(self.subprotocols["purify_B1"], Signals.SUCCESS)) )
            
            signal_A = self.subprotocols["purify_A1"].get_signal_result(Signals.SUCCESS, self)
            signal_B = self.subprotocols["purify_B1"].get_signal_result(Signals.SUCCESS, self)
            
            if (self._nodes._data['A'].qmemory.mem_positions[signal_A]._qubit.qstate==None or self._nodes._data['B'].qmemory.mem_positions[signal_B]._qubit.qstate==None):
                yield self.await_timer(duration=self.source_delay)
                self.send_signal(Signals.FAIL)

            else:
                result = {
                    "pos_A": signal_A,
                    "pos_B": signal_B,
                    "time": sim_time() - start_time,
                    "pairs": self.subprotocols["entangle_A"].entangled_pairs,
                }
                self.send_signal(Signals.SUCCESS, result)

        
class DistilThrice(LocalProtocol):
    def __init__(self, node_a, node_b, inputmempos_a, inputmempos_b, num_runs, con_name, qs_no, source_delay=5.5*10e3 ):
        super().__init__(nodes={"A": node_a, "B": node_b}, name="Distillation example")
        nodeAltr= node_a.name[5]
        qs2 = str(int(qs_no)+1)
        qs3 = str(int(qs_no)+2)
        qs4 = str(int(qs_no)+3)
        self.num_runs = num_runs
        self.source_delay =source_delay

        # Initialise sub-protocols
        self.add_subprotocol(EntangleBase(node=node_a, role="source", input_mem_pos=inputmempos_a,num_pairs=2, 
                            name="entangle_A", qs_name= f"QSource_{nodeAltr}{qs_no}", qs_name1 = f"QSource_{nodeAltr}{qs2}"))
        self.add_subprotocol(EntangleBase(node=node_b, role="receiver", input_mem_pos=inputmempos_b, num_pairs=2,
                          name="entangle_B", qs_name = f"QSource_{nodeAltr}{qs_no}", qs_name1 = f"QSource_{nodeAltr}{qs2}"))

        self.add_subprotocol(EntangleNodes(node=node_a, role="source", input_mem_pos=2,
                                           num_pairs=1, name="entangle_A1", qs_name= f"QSource_{nodeAltr}{qs3}"))
        self.add_subprotocol(EntangleNodes(node=node_b, role="receiver", input_mem_pos=2, 
                                           num_pairs=1,name="entangle_B1", qs_name = f"QSource_{nodeAltr}{qs3}"))
        
        self.add_subprotocol(EntangleNodes(node=node_a, role="source", input_mem_pos=3,
                                           num_pairs=1, name="entangle_A2", qs_name= f"QSource_{nodeAltr}{qs4}"))
        self.add_subprotocol(EntangleNodes(node=node_b, role="receiver", input_mem_pos=3, 
                                           num_pairs=1,name="entangle_B2", qs_name= f"QSource_{nodeAltr}{qs4}"))

        self.add_subprotocol(Distil(node_a, node_a.get_conn_port(node_b.ID, con_name), "A",
                                     name="purify_A", con_num = con_name, imp = inputmempos_a))
        self.add_subprotocol(Distil(node_b, node_b.get_conn_port(node_a.ID, con_name), "B",
                                     name="purify_B",con_num = con_name, imp = inputmempos_b))
        
        self.add_subprotocol(DistilFurther(node_a, node_a.get_conn_port(node_b.ID, "AB1"), "A",
                                     name="purify_A1", con_num = "AB1", distilled_pos = 1, imp = inputmempos_a + 2))
        self.add_subprotocol(DistilFurther(node_b, node_b.get_conn_port(node_a.ID, "AB1"), "B",
                                     name="purify_B1",con_num = "AB1", distilled_pos = 1, imp = inputmempos_b + 2))
        self.add_subprotocol(DistilFurther(node_a, node_a.get_conn_port(node_b.ID, "AB2"), "A",
                                     name="purify_A2", con_num = "AB2", distilled_pos = 1, imp= inputmempos_a+3))
        self.add_subprotocol(DistilFurther(node_b, node_b.get_conn_port(node_a.ID, "AB2"), "B",
                                     name="purify_B2",con_num = "AB2", distilled_pos = 1, imp= inputmempos_b+3))

        self.subprotocols["purify_A"].start_expression = (
            self.subprotocols["purify_A"].await_signal(self.subprotocols["entangle_A"],
                                                       Signals.SUCCESS))
        self.subprotocols["purify_B"].start_expression = (
            self.subprotocols["purify_B"].await_signal(self.subprotocols["entangle_B"],
                                                       Signals.SUCCESS))
        self.subprotocols["purify_A"].start_expression_qb2 = (
            self.subprotocols["purify_A"].await_signal(self.subprotocols["entangle_A"],
                                                       Signals.FINISHED))
        self.subprotocols["purify_B"].start_expression_qb2 = (
            self.subprotocols["purify_B"].await_signal(self.subprotocols["entangle_B"],
                                                       Signals.FINISHED))
        self.subprotocols["purify_A1"].start_expression = (
            self.subprotocols["purify_A1"].await_signal(self.subprotocols["entangle_A1"],
                                                       Signals.SUCCESS))
        self.subprotocols["purify_B1"].start_expression = (
            self.subprotocols["purify_B1"].await_signal(self.subprotocols["entangle_B1"],
                                                       Signals.SUCCESS))
        self.subprotocols["purify_A2"].start_expression = (
            self.subprotocols["purify_A2"].await_signal(self.subprotocols["entangle_A2"],
                                                       Signals.SUCCESS))
        self.subprotocols["purify_B2"].start_expression = (
            self.subprotocols["purify_B2"].await_signal(self.subprotocols["entangle_B2"],
                                                       Signals.SUCCESS))
        
        start_expr_ent_A = (self.subprotocols["entangle_A"].await_signal(
                            self.subprotocols["purify_A"], Signals.FAIL)|
                            self.subprotocols["entangle_A"].await_signal(
                            self.subprotocols["purify_A1"], Signals.FAIL)|
                            self.subprotocols["entangle_A"].await_signal(
                            self.subprotocols["purify_A2"], Signals.FAIL) |
                            self.subprotocols["entangle_A"].await_signal(
                                self, Signals.WAITING))
        
        start_expr_ent_B = (self.subprotocols["entangle_B"].await_signal(
                            self.subprotocols["purify_B"], Signals.FAIL) |
                            self.subprotocols["entangle_B"].await_signal(
                            self.subprotocols["purify_B1"], Signals.FAIL)|
                            self.subprotocols["entangle_B"].await_signal(
                            self.subprotocols["purify_B2"], Signals.FAIL) |
                            self.subprotocols["entangle_B"].await_signal(
                                self, Signals.WAITING))
        
        self.subprotocols["entangle_A"].start_expression = start_expr_ent_A
        self.subprotocols["entangle_B"].start_expression = start_expr_ent_B
        
        start_expr_ent_A1 = (self.await_signal(
                             self.subprotocols["purify_A"], Signals.SUCCESS)& 
                             self.await_signal(
                             self.subprotocols["purify_B"], Signals.SUCCESS))   
        self.subprotocols["entangle_A1"].start_expression = start_expr_ent_A1
        self.subprotocols["entangle_B1"].start_expression = start_expr_ent_A1


        start_expr_ent_A2 = (self.await_signal(
                             self.subprotocols["purify_A1"], Signals.SUCCESS)& 
                             self.await_signal(
                             self.subprotocols["purify_B1"], Signals.SUCCESS))
        self.subprotocols["entangle_A2"].start_expression = start_expr_ent_A2
        self.subprotocols["entangle_B2"].start_expression = start_expr_ent_A2


    def run(self):
        self.start_subprotocols()
        for i in range(self.num_runs):
            
            start_time = sim_time()
            self.subprotocols["entangle_A"].entangled_pairs = 0
            self.subprotocols["entangle_A1"].entangled_pairs = 0
            self.subprotocols["entangle_A2"].entangled_pairs = 0
            self.send_signal(Signals.WAITING)

            yield ( (self.await_signal(self.subprotocols["purify_A2"], Signals.SUCCESS) & self.await_signal(self.subprotocols["purify_B2"], Signals.SUCCESS)) )

            signal_A = self.subprotocols["purify_A2"].get_signal_result(Signals.SUCCESS, self)
            signal_B = self.subprotocols["purify_B2"].get_signal_result(Signals.SUCCESS, self)
            
            if (self._nodes._data['A'].qmemory.mem_positions[signal_A]._qubit.qstate==None or self._nodes._data['B'].qmemory.mem_positions[signal_B]._qubit.qstate==None):
                yield self.await_timer(duration=self.source_delay)
                self.send_signal(Signals.FAIL)
            else:
                result = {
                    "pos_A": signal_A,
                    "pos_B": signal_B,
                    "time": sim_time() - start_time,
                    "pairs": self.subprotocols["entangle_A"].entangled_pairs,
                }
                self.send_signal(Signals.SUCCESS, result)
        
def add_connection(network, nodea, nodeb, src_conns, dest_conns, con_num, 
                   node_distance=1, p_loss_node=2, p_loss_fibre=0.16):
    
    nodeAltr, nodeBltr= nodea.name[5], nodeb.name[5]
    alwr, blwr = nodeAltr.lower(), nodeBltr.lower()
    
    conn_cchannel = DirectConnection(
        f"CChannelConn_{nodeAltr}{nodeBltr}",
        ClassicalChannel(f"CChannel_{nodeAltr}->{nodeBltr}_{con_num*2}", length=node_distance,
                         models={"delay_model": FibreDelayModel(c=200e3)}),
        ClassicalChannel(f"CChannel_{nodeBltr}->{nodeAltr}_{con_num*2}", length=node_distance,
                         models={"delay_model": FibreDelayModel(c=200e3)}))    
    network.add_connection(nodea, nodeb, connection=conn_cchannel, label = f"{nodeAltr}{nodeBltr}{con_num}")
    
    qchannel = QuantumChannel(f"QuantumChannel_{nodeAltr}->{nodeBltr}_{con_num*2}", length=node_distance)
    qchannel.models['quantum_loss_model'] = CustomLossModel( p_loss_node, p_loss_fibre)
    # qchannel.models['quantum_noise_model'] = CustomDecoherenceModel()
    qchannel.models["delay_model"] = FibreDelayModel(c=200e3)
    port_name_a, port_name_b = network.add_connection(
        nodea, nodeb, channel_to=qchannel, label=f"{alwr}{blwr}{con_num*2}")
    
    qchannel1 = QuantumChannel(f"QuantumChannel_{nodeAltr}->{nodeBltr}_{con_num*2+1}", length=node_distance)
    qchannel1.models['quantum_loss_model'] = CustomLossModel( p_loss_node, p_loss_fibre)
    # qchannel1.models['quantum_noise_model'] = CustomDecoherenceModel()
    qchannel1.models["delay_model"] = FibreDelayModel(c=200e3)
    port_name_a1, port_name_b1 = network.add_connection(
        nodea, nodeb, channel_to=qchannel1, label=f"{alwr}{blwr}{con_num*2+1}")

    aport, bport= src_conns*2, dest_conns*2
    qmem_a_num, qmem_b_num  = 0, 0
    
    nodea.subcomponents[f"QSource_{nodeAltr}{src_conns*2}"].ports["qout1"].forward_output(
        nodea.ports[port_name_a])
    nodea.subcomponents[f"QSource_{nodeAltr}{src_conns*2}"].ports["qout0"].connect(
        nodea.subcomponents[f"QuantumMemory_{nodeAltr}{qmem_a_num}"].ports[f"qin{aport}"])
    
    nodea.subcomponents[f"QSource_{nodeAltr}{src_conns*2+1}"].ports["qout1"].forward_output(
        nodea.ports[port_name_a1])
    nodea.subcomponents[f"QSource_{nodeAltr}{src_conns*2+1}"].ports["qout0"].connect(
        nodea.subcomponents[f"QuantumMemory_{nodeAltr}{qmem_a_num}"].ports[f"qin{aport+1}"])
    
    nodeb.ports[port_name_b].forward_input(nodeb.subcomponents[f"QuantumMemory_{nodeBltr}{qmem_b_num}"].ports[f"qin{bport}"])
    nodeb.ports[port_name_b1].forward_input(nodeb.subcomponents[f"QuantumMemory_{nodeBltr}{qmem_b_num}"].ports[f"qin{bport+1}"])

def sim_setup(node_a, node_b, num_runs, imp_src, imp_dest, con_name="", qs_no ="", network_load=0, dist=1, K=1):
    """sim setup for purification protocols.

    """
    dist_example = DistilSetup(node_a, node_b, imp_src, imp_dest, num_runs=num_runs, con_name= con_name, qs_no=qs_no)

    def record_run(evexpr):
        # Callback that collects data each run
        protocol = evexpr.triggered_events[-1].source
        result = protocol.get_signal_result(Signals.SUCCESS)
        # Record fidelity
        q_A, = node_a.qmemory.pop(positions=[result["pos_A"]])
        q_B, = node_b.qmemory.pop(positions=[result["pos_B"]])

        f2 = qapi.fidelity([q_A, q_B], ks.b01, squared=True)
        return {"F2": f2,  "time": result["time"],  
                "Node_src": node_a.name[5], "Node_dest": node_b.name[5],  
                "dist": dist, "network_load": network_load, "K" : K}

    dc = DataCollector(record_run, include_time_stamp=False,
                       include_entity_name=False)
    dc.collect_on(pd.EventExpression(source=dist_example,
                                     event_type=Signals.SUCCESS.value))
    return dist_example, dc


def rem_connection(network, nodea, nodeb, con_num):
    nodeAltr, nodeBltr= nodea.name[5], nodeb.name[5]
    alwr, blwr = nodeAltr.lower(), nodeBltr.lower()
    conn = network.get_connection(nodea, nodeb, label = f"{nodeAltr}{nodeBltr}{con_num}")
    qconn = network.get_connection(nodea, nodeb, label = f"{alwr}{blwr}{con_num}")
    network.remove_connection(conn)
    network.remove_connection(qconn)
                   



def find_channel(shortest_paths, available_channels, K, num_channels):
    channel = None
    blocked = False
    #iterate through channels
    for k in range(K):
        for j in range (num_channels):
        #We have tranversed a path with the same wavelength available the whole way
            blocked = False
            for i in range(len(shortest_paths[k])-1):
                if(available_channels[shortest_paths[k][i]][shortest_paths[k][i+1]][j]==0):
                    channel = j
                else:
                    blocked = True
                    channel = None

            if ((channel!=None)&(blocked==False)):
                for i in range(len(shortest_paths[k])-1):
                    available_channels[shortest_paths[k][i]][shortest_paths[k][i+1]][j]=1
                return channel, shortest_paths[k]

    
    print("No channel available!")
    return -1, -1

def get_name_and_dist(path):
    src = chr(path[0] + 65)
    dest = chr(path[-1] + 65)
    dist = len(path)
    return src, dest, dist 

# def generate_users(lam=5):
#     b = np.random.poisson(lam)
#     sourcelist= np.random.randint(6, size=(5))
#     destlist= np.random.randint(6, size=(5))
#     return sourcelist, destlist 

def generate_users(G, lam=0.5):
    sourcelist, destlist = [],[]

    for i in range(len(G)):
        source = np.random.poisson(lam)
        for j in range(source):
            sourcelist.append(i)
            destlist.append(random.choice([k for k in range(0,len(G)) if k not in  [i]]))

    
    return sourcelist, destlist


def create_random_connections(G, network, channels, total_cons, total_links, dist_examples, dataframes, max_links, K, num_runs, q_mem_size, num_channels, node_dist, blocking, l=0.5):
    sources, dests = generate_users(G, l)
    mem_blocked = 0
    wavelength_blocked = 0
    for i in range(len(sources)):
        if(total_cons[sources[i]]*2>=q_mem_size-2):
            dc = pandas.DataFrame({"Node_src":sources[i], "Mem full":True, "network_load" : round(np.sum(channels)/max_links, 2)}, index =[0])
            blocking.append(dc)
            print(f"Node_{sources[i]} full")
            mem_blocked+=1
            continue
        
        elif(total_cons[dests[i]]*2>=q_mem_size):
            dc = pandas.DataFrame({"Node_src":dests[i], "Mem full":True, "network_load" : round(np.sum(channels)/max_links, 2)}, index =[0])
            blocking.append(dc)
            print(f"Node_{dests[i]} full")
            mem_blocked+=1
            continue

        shortest_paths = k_shortest_paths(G, str(sources[i]), str(dests[i]), K)
        wavelength, path = find_channel(shortest_paths, channels, len(shortest_paths), num_channels)
        if (wavelength!=-1):
            src, dest, num_nodes_traversed = get_name_and_dist(path)
            d = num_nodes_traversed*node_dist-1
            con_num=total_links[path[0]][path[-1]]
            add_connection(network, network.get_node(f"node_{src}"), network.get_node(f"node_{dest}"), p_loss_node = num_nodes_traversed*2
                        , src_conns=total_cons[path[0]], dest_conns=total_cons[path[-1]], con_num=con_num, node_distance=(num_nodes_traversed-1)*node_dist)
                #rem_connection(network, network.get_node(f"node_{src}"), network.get_node(f"node_{dest}"), con_num)
            dist_protocol, dc = sim_setup(network.get_node(f"node_{src}"),network.get_node(f"node_{dest}"), imp_src=total_cons[path[0]]*2,
                            imp_dest= total_cons[path[-1]]*2, con_name = f"{src}{dest}{con_num}",num_runs=num_runs,qs_no = f"{total_cons[path[0]]*2}"
                            ,network_load = round(np.sum(channels)/max_links, 2), dist=num_nodes_traversed-1, K=K)
 
            dist_protocol.start()
            total_links[path[0]][path[-1]]+=1
            total_cons[path[0]]+=1
            total_cons[path[-1]]+=1
            dist_examples.append(dist_protocol)
            dataframes.append(dc)
            ns.sim_run(duration=31)
        else:
            wavelength_blocked+=1
            dc = pandas.DataFrame({"Node_src":sources[i], "Node_dest": dests[i], "blocked":True, "network_load" : round(np.sum(channels)/max_links, 2)}, index =[0])
            blocking.append(dc)
            ns.sim_run(duration=31)

    if (mem_blocked >= len(sources)):   #if for all pairs generated this round, no mems found return 
        return True
    if(wavelength_blocked>=len(sources)): #if no wavelengths available for all pairs generated this round,  return 
        return True

    ns.sim_run(duration=300029)
    


def dijkstra_shortest_paths(graph, source):
    """
    Computes the shortest paths from the source node to all other nodes in the graph using Dijkstra's algorithm.
    :param graph: The graph represented as a dictionary of dictionaries. Each key is a node and the corresponding value
                  is a dictionary that maps its neighbors to their edge weights.
    :param source: The source node from which to compute the shortest paths.
    :return: A dictionary of the shortest distances from the source node to all other nodes in the graph.
    """
    distances = {source: 0}
    heap = [(0, source)]
    while heap:
        (distance, node) = heapq.heappop(heap)
        if node in graph:
            for neighbor, weight in graph[node].items():
                new_distance = distance + weight
                if neighbor not in distances or new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    heapq.heappush(heap, (new_distance, neighbor))
    return distances

def k_shortest_paths(graph, source, target, k):
    """
    Computes the k shortest paths from the source node to the target node in the graph.
    :param graph: The graph represented as a dictionary of dictionaries. Each key is a node and the corresponding value
                  is a dictionary that maps its neighbors to their edge weights.
    :param source: The source node from which to compute the shortest paths.
    :param target: The target node to which to compute the shortest paths.
    :param k: The number of shortest paths to compute.
    :return: A list of the k shortest paths from the source node to the target node in the graph.
    """
    paths = []
    distances = dijkstra_shortest_paths(graph, source)
    heap = [(0, [source])]
    while heap and len(paths) < k:
        (distance, path) = heapq.heappop(heap)
        node = path[-1]
        if node == target:
            paths.append(list(map(int, path)))
        if node in graph:
            for neighbor, weight in graph[node].items():
                if neighbor not in path:
                    new_path = path + [neighbor]
                    new_distance = distance + weight
                    heapq.heappush(heap, (new_distance, new_path))
    return paths

def t1t2_plot(num_iters=50):
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(ncols=2)
    #(3600*1e9,1.46e9, 1), (3600*1e9,1.46e9, 2), (3600*1e9,1.46e9, 3)
    
    for t1, t2, k in  (2.68e6, 3.3e3, 1), (2.68e6, 3.3e3, 2), (2.68e6, 3.3e3, 3):
        data = pandas.DataFrame()
        time_data = pandas.DataFrame()
        for d in [0.1, 0.2, 0.4, 0.8, 1.6]:
            res = six_node_network(t1 = t1, t2=t2, q_mem_size=100, node_distance=d, num_channels = 10, num_runs=num_iters, dist_runs = 1, k =k)
            time_data[d] = pow(res['time'], -1)*1e9
            data[d] = res['F2']        
        # For errorbars we use the standard error of the mean (sem)
        data = data.agg(['mean', 'sem']).T.rename(columns={'mean': 'fidelity'})
        time_data = time_data.agg(['mean', 'sem']).T.rename(columns={'mean': 'time'})
        
        data.plot(y='fidelity', yerr='sem', label=f"T1={t1}, T2={t2}, K:{k}", ax=axes[0])
        time_data.plot(y='time', yerr='sem', label=f"T1={t1}, T2={t2}, K:{k}", ax=axes[1])
 
    plt.title("Fidelity for different T1, T2, K-shortest paths")

    fig.suptitle("Comparison of nuclear and electron spins")
    axes[0].set_xlabel("Total Distance of link (km)")
    axes[0].set_ylabel("Average fidelity on success")
    axes[0].set_title("Avg fidelity")
    axes[1].set_xlabel("Total Distance of link (km)")
    axes[1].set_ylabel("ebit rate (ebits/s)")
    axes[1].set_title("Average ebit rate")

    plt.show()

def runandsave():
    ns.sim_reset()
    df3 = pandas.DataFrame()
    df2 = pandas.DataFrame()
    df1 = pandas.DataFrame()

    np.random.seed(50)
    for i in range(50):
        res = six_node_network(3, 3, 200, 10,node_distance=0.5, t1 = 2.68e6, t2= 3.3e3)
        df3 = pandas.concat([df3,res], ignore_index=True)
        
    print("done k1")
    df3.to_csv("~/qproject/k1_d=0.5._10chan_tsmall.csv", index=False )

    np.random.seed(50)
    for i in range(50):
        res = six_node_network(2, 3, 200, 10, node_distance=0.5, t1 = 2.68e6, t2= 3.3e3)
        df2 = pandas.concat([df2,res], ignore_index=True)
    print("done k2")
    df2.to_csv("~/qproject/k2_d=0.5._10chan_tsmall.csv", index=False)

    np.random.seed(50)
    for i in range(50):
        res = six_node_network(1, 3, 200, 10, node_distance=0.5, t1 = 2.68e6, t2= 3.3e3)
        df1 = pandas.concat([df1,res], ignore_index=True)
    df1.to_csv("~/qproject/k3_d=0.5._10chan_tsmall.csv", index=False)

    grouped1 = df1.groupby('network_load')['F2'].mean()
    grouped2 = df2.groupby('network_load')['F2'].mean()
    grouped3 = df3.groupby('network_load')['F2'].mean()

def mem_blocking(num_iters=3):
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(ncols=1)
    #(3600*1e9,1.46e9, 1), (3600*1e9,1.46e9, 2), (3600*1e9,1.46e9, 3)
    
    for mems in [10,20,40,80]:
        
        data = pandas.DataFrame()
        time_data = pandas.DataFrame()
       
        for channels in [1,3,5,10,15,20]:
            # np.random.seed(50)
            # res = pandas.DataFrame()
            # for i in range(10):
            #     res = pandas.concat([res,six_node_network(q_mem_size=mems, node_distance=1, 
            #                         num_channels = channels, num_runs=num_iters, dist_runs = 1, k =3)], ignore_index=True)
            # time_data[channels] = pow(res['time'], -1)*1e9
            # data[channels] = res['F2']        
        
            # res.to_csv(f"~/qproject/repeatedruns/mem{mems}chan{channels}")
            res =  pandas.read_csv(f"~/qproject/repeatedruns/mem{mems}chan{channels}")
            data[channels]=res['F2'] 
            time_data[channels] = pow(res['time'], -1)*1e9
        # For errorbars we use the standard error of the mean (sem)
        data = data.agg(['mean', 'sem']).T.rename(columns={'mean': 'fidelity'})
        time_data = time_data.agg(['mean', 'sem']).T.rename(columns={'mean': 'time'})
        # data.plot(y='fidelity', yerr='sem', label=f"Mem size = {mems}", ax=axes[0])
        time_data.plot(y='time', yerr='sem', label=f"Mem size = {mems}", ax=axes)
        print (f"done {channels}")
        
 
    # plt.title("Qmemory sizes vs number of channels")

    #fig.suptitle("Qmemory sizes vs number of channels, Dynamical Decoupling, K=3")
    axes.set_xlabel("Number of channels")
    #axes[0].set_ylabel("Average fidelity on success")
    axes.set_ylabel("ebit rate (ebits/s)")
    axes.set_title("Qmemory sizes vs number of channels, Dynamical Decoupling, K=3")
    # axes[1].set_xlabel("Number of channels")
    # axes[1].set_ylabel("ebit rate (ebits/s)")
    # axes[1].set_title("Average ebit rate")

    plt.show()

def six_node_network(k, num_runs, q_mem_size, num_channels, t1= 3600 * 1e9, t2=1.46e9, node_distance=1, sf=0.994, dist_runs =1, l=0.5):
    ns.sim_reset()
    network = example_network_setup(qprocessor_positions=q_mem_size, t1 = t1, t2 =t2, source_fidelity_sq=sf)

    total_cons = [0,0,0,0,0,0]
    total_links =  [[0, 0, 0, 0, 0 ,0],
                    [0, 0, 0, 0, 0 ,0],
                    [0, 0, 0, 0, 0 ,0],
                    [0, 0, 0, 0, 0 ,0],
                    [0, 0, 0, 0, 0 ,0],
                    [0, 0, 0, 0, 0 ,0]]
    

    G = {       '0': {'1': 1, '2': 1},
                '1': {'0': 1, '2': 1, '3': 1},
                '2': {'0': 1, '1': 1, '4': 1},
                '3': {'1': 1, '4': 1, '5': 1},
                '4': {'2': 1, '3': 1, '5': 1},
                '5': {'3': 1, '4': 1},         }
      
    channels = np.zeros((len(G), 6, num_channels))
    total_sum = 0
    for row in G:
            total_sum += sum(G[row].values())
    max_links = total_sum*num_channels
    dataframes, dist_examples, blocking = [],[],[]
    blocked_count = 0
   
    while (np.sum(channels)/max_links< 0.7) and (sum(total_cons)<0.7*q_mem_size*6) and (blocked_count<4):
        blocked = create_random_connections(G, network, channels, total_cons, total_links, dist_examples, dataframes, max_links, k, num_runs, q_mem_size, num_channels, node_distance, blocking, l)
        if(blocked == True):
            blocked_count +=1  #if no pairs could connect last round, system is getting pretty blocked
        else:                   #if 4 times in a row no mems or wavelengths available, we're done
            blocked_count = 0
    ns.sim_run()
    #network.nodes._map._data['node_A'].qmemory.unused_positions
    results = pandas.DataFrame()
    for i in range(len(dataframes)):
        results = pandas.concat([results,dataframes[i].dataframe], ignore_index=True)
    for i in range(len(blocking)):
        results = pandas.concat([results,blocking[i]], ignore_index=True, axis = 0)

    del dataframes

    return results

def node_bars():
    df = pandas.read_csv("~/qproject/k1_d=0.5._10chan_tsmall.csv")
    df =df[df["network_load"]<0.55]
    nodecounts  = df["Node_src"].value_counts()
    nodecounts  =nodecounts+ df["Node_dest"].value_counts()
    
    df1 = pandas.read_csv("~/qproject/k2_d=0.5._10chan_tsmall.csv")
    df1 =df1[df1["network_load"]<0.55]
    nodecounts1 = df1["Node_src"].value_counts()
    nodecounts1  =nodecounts1+ df1["Node_dest"].value_counts()

    df2 = pandas.read_csv("~/qproject/k3_d=0.5._10chan_tsmall.csv")
    df2 =df2[df2["network_load"]<0.55]
    nodecounts2  = df2["Node_src"].value_counts()
    nodecounts2  =nodecounts2+ df2["Node_dest"].value_counts()
    # nodecounts2.sort_index(ascending=True, inplace=True)

    width = 0.25
    N = 6
    ind = np.arange(N)

    bar1 = plt.bar(ind, nodecounts, width, color = 'r')
    bar2 = plt.bar(ind+width, nodecounts1, width, color='g')
    bar3 = plt.bar(ind+width*2, nodecounts2, width, color = 'b')
    
    plt.title("Total link usage nework load < 0.55")
    plt.xlabel("Nodes")
    plt.ylabel('total node usage')
    plt.xticks(ind+width,['A', 'B', 'C', 'D','E','F'])
    plt.legend((bar1,bar2,bar3), ('K=1', 'K=2', 'K=3') )
    plt.show()

def read_and_plot(path):
    fig, ax = plt.subplots()
    for k in [1,2,3]:
        df = pandas.read_csv(f"~/qproject/6node/10channels/k{k}d=0.5.csv")
        df = df.drop(columns=df.columns[0])
        df["time"]=pow(df['time'], -1)*1e9
        grouped1 = df.groupby("network_load")["time"].agg(['mean', 'sem'])
        grouped1.plot(y='mean', yerr='sem', label=f"K={k}", ax=ax)
    plt.title("Average ebit rate 6 node network - Nuclear Spins")
    plt.xlabel('Network load')
    plt.ylabel('ebit rate')
    plt.show()
  
if __name__ == "__main__":
    
    # ns.sim_reset()
    # runandsave()
    # df =  six_node_network(3, 100, 10, 10)
    # df["time"]=pow(df['time'], -1)*1e9
    # # grouped1 = df.groupby("network_load")["time"].agg(['mean', 'sem'])
    # grouped1 = df.groupby("network_load")["dist"].agg(['mean', 'sem'])
    # grouped1.plot(y='mean', yerr='sem', label=f"K={3}")
    # plt.title("Average ebit rate 6 node network - Nuclear Spins")
    # plt.xlabel('Network load')
    # plt.ylabel('ebit rate')
    # plt.show()
    mem_blocking()
    









