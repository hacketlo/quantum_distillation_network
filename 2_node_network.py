
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

def example_network_setup(source_delay=5.5*10e3, source_fidelity_sq=0.994,  p_loss_node=2, p_loss_fibre=0.16,
                          node_distance=1, qprocessor_positions = 6, t1= 3600 * 1e9, t2=1.46e9, physical = True):

    network = Network("purify_network")

    _INSTR_Rx = IGate("Rx_gate", ops.create_rotation_op(np.pi / 2, (1, 0, 0)))
    _INSTR_RxC = IGate("RxC_gate", ops.create_rotation_op(np.pi / 2, (1, 0, 0), conjugate=True))
    #memory_noise_models=T1T2NoiseModel(3600*1e9, 1.46e9)

    node_a, node_b = network.add_nodes(["node_A", "node_B"])
    if (physical == True):
        node_a.add_subcomponent(QuantumProcessor(
            "QuantumMemory_A0", num_positions=qprocessor_positions, fallback_to_nonphysical=True, memory_noise_models=T1T2NoiseModel(t1, t2), phy_instructions =[
            PhysicalInstruction(_INSTR_Rx, duration=5, parallel=True),
            PhysicalInstruction(_INSTR_RxC, duration=5, parallel=True),
            PhysicalInstruction(INSTR_CNOT, duration=20, parallel=True),
            PhysicalInstruction(INSTR_MEASURE, duration=35, parallel=True),
            PhysicalInstruction(INSTR_SWAP, duration=20, parallel=True)]   ))

        node_b.add_subcomponent(QuantumProcessor(
            "QuantumMemory_B0", num_positions=qprocessor_positions, fallback_to_nonphysical=True, memory_noise_models=T1T2NoiseModel(t1, t2), phy_instructions =[
            PhysicalInstruction(_INSTR_Rx, duration=5, parallel=True),
            PhysicalInstruction(_INSTR_RxC, duration=5, parallel=True),
            PhysicalInstruction(INSTR_CNOT, duration=20, parallel=True),
            PhysicalInstruction(INSTR_MEASURE, duration=35, parallel=True),
            PhysicalInstruction(INSTR_SWAP, duration=20, parallel=True)]   ))
    else:
        node_a.add_subcomponent(QuantumProcessor(
            "QuantumMemory_A0", num_positions=qprocessor_positions, fallback_to_nonphysical=True, memory_noise_models=T1T2NoiseModel(t1, t2)  ))
        node_b.add_subcomponent(QuantumProcessor(
            "QuantumMemory_B0", num_positions=qprocessor_positions, fallback_to_nonphysical=True, memory_noise_models=T1T2NoiseModel(t1, t2) ))
    

    state_sampler = StateSampler(
        [ks.b01, ks.s00],
        probabilities=[source_fidelity_sq, (1 - source_fidelity_sq)])
    
    for i in range(qprocessor_positions):
        node_a.add_subcomponent(QSource(
            f"QSource_A{i}", state_sampler=state_sampler,
            models={"emission_delay_model": FixedDelayModel(delay=source_delay)},
            num_ports=2, status=SourceStatus.EXTERNAL))
    
    for i in range(qprocessor_positions*2):
        node_b.add_subcomponent(QSource(
            f"QSource_B{i}", state_sampler=state_sampler,
            models={"emission_delay_model": FixedDelayModel(delay=source_delay)},
            num_ports=2, status=SourceStatus.EXTERNAL))
  
    conn_cchannel_ab = DirectConnection(
        "CChannelConn_AB_0",
        ClassicalChannel("CChannel_A->B_0", length=node_distance,
                         models={"delay_model": FibreDelayModel(c=200e3)}),
        ClassicalChannel("CChannel_B->A_0", length=node_distance,
                         models={"delay_model": FibreDelayModel(c=200e3)}))
    network.add_connection(node_a, node_b, connection=conn_cchannel_ab, label = "AB0")
 
    qchannel_ab = QuantumChannel("QuantumChannelA->B_0", length=node_distance)
    # qchannel_ab.models['quantum_noise_model'] = CustomDecoherenceModel()

    qchannel_ab.models['quantum_loss_model'] = CustomLossModel( p_loss_node, p_loss_fibre)
    qchannel_ab.models["delay_model"] = FibreDelayModel(c=200e3) #km/s
    port_name_a, port_name_b = network.add_connection(
        node_a, node_b, channel_to=qchannel_ab, label="ab0")
    
    qchannel1 = QuantumChannel(f"QuantumChannel_A->B_1", length=node_distance)
    # qchannel1.models['quantum_noise_model'] = CustomDecoherenceModel()

    qchannel1.models['quantum_loss_model'] = CustomLossModel( p_loss_node, p_loss_fibre)
    qchannel1.models["delay_model"] = FibreDelayModel(c=200e3)
    port_name_a1, port_name_b1 = network.add_connection(
        node_a, node_b, channel_to=qchannel1, label=f"ab1")
    
    #CHANNEL AB0
    # Link source ports:
    node_a.subcomponents["QSource_A0"].ports["qout1"].forward_output(
        node_a.ports[port_name_a])
    node_a.subcomponents["QSource_A0"].ports["qout0"].connect(
        node_a.qmemory.ports["qin0"])
    node_a.subcomponents["QSource_A1"].ports["qout1"].forward_output(
        node_a.ports[port_name_a1])
    node_a.subcomponents["QSource_A1"].ports["qout0"].connect(
        node_a.qmemory.ports["qin1"])
    
    # Link dest ports:
    node_b.ports[port_name_b].forward_input(node_b.qmemory.ports["qin0"])
    node_b.ports[port_name_b1].forward_input(node_b.qmemory.ports["qin1"])
    
    return network


class DistilFurther(NodeProtocol):

    # set basis change operators for local DEJMPS step
    _INSTR_Rx = IGate("Rx_gate", ops.create_rotation_op(np.pi / 2, (1, 0, 0)))
    _INSTR_RxC = IGate("RxC_gate", ops.create_rotation_op(np.pi / 2, (1, 0, 0), conjugate=True))

    def __init__(self, node, port, role, start_expression=None, msg_header="distil", name=None, con_num = 0, distilled_pos = 1, imp = 0, phys = True):
        if role.upper() not in ["A", "B"]:
            raise ValueError
        conj_rotation = role.upper() == "B"
        if not isinstance(port, Port):
            raise ValueError("{} is not a Port".format(port))
        name = name if name else "DistilNode({}, {}, {})".format(node.name, port.name, con_num)
        super().__init__(node, name=name)
        self.port = port
        self.instance_id = con_num
        self.physical = phys
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
        prog.apply(INSTR_ROT, [q1], physical=self.physical)
        prog.apply(INSTR_ROT, [q2], physical=self.physical)
        prog.apply(INSTR_CNOT, [q1, q2], physical=self.physical)
        prog.apply(INSTR_MEASURE, q2, output_key="m", inplace=False, physical=self.physical)
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

    def __init__(self, node, port, role, start_expression=None, start_expression_qb2=None, msg_header="distil", name=None, con_num = 0, imp = 0, phys =True):
        if role.upper() not in ["A", "B"]:
            raise ValueError
        conj_rotation = role.upper() == "B"
        if not isinstance(port, Port):
            raise ValueError("{} is not a Port".format(port))
        name = name if name else "DistilNode({}, {}, {})".format(node.name, port.name, con_num)
        super().__init__(node, name=name)
        self.port = port
        self.physical = phys
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
        prog.apply(INSTR_ROT, [q1], physical = self.physical)
        prog.apply(INSTR_ROT, [q2], physical=self.physical)
        prog.apply(INSTR_CNOT, [q1, q2], physical=self.physical)
        prog.apply(INSTR_MEASURE, q2, output_key="m", inplace=False, physical=self.physical)
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

    def __init__(self, node_a, node_b, inputmempos_a, inputmempos_b, num_runs, con_name, qs_no, source_delay=5.5*10e3):
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

    def __init__(self, node_a, node_b, inputmempos_a, inputmempos_b, num_runs, con_name, qs_no, source_delay=5.5*10e3,phys =True ) :
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
                                     name="purify_A", con_num = con_name, phys =phys))
        self.add_subprotocol(Distil(node_b, node_b.get_conn_port(node_a.ID, con_name), "B",
                                     name="purify_B",con_num = con_name, phys = phys))
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

    def __init__(self, node_a, node_b, inputmempos_a, inputmempos_b, num_runs, con_name, qs_no, source_delay=5.5*10e3, phys =True):
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
                                     name="purify_A", con_num = con_name, imp = inputmempos_a, phys =phys))
        self.add_subprotocol(Distil(node_b, node_b.get_conn_port(node_a.ID, con_name), "B",
                                     name="purify_B",con_num = con_name, imp = inputmempos_b, phys = phys))
        
        self.add_subprotocol(DistilFurther(node_a, node_a.get_conn_port(node_b.ID, "AB1"), "A",
                                     name="purify_A1", con_num = "AB1", distilled_pos = 1, imp = inputmempos_a + 2, phys = phys))
        self.add_subprotocol(DistilFurther(node_b, node_b.get_conn_port(node_a.ID, "AB1"), "B",
                                     name="purify_B1",con_num = "AB1", distilled_pos = 1, imp = inputmempos_b + 2, phys = phys))

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
    def __init__(self, node_a, node_b, inputmempos_a, inputmempos_b, num_runs, con_name, qs_no, source_delay=5.5*10e3, phys = True ):
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
                                     name="purify_A", con_num = con_name, imp = inputmempos_a, phys = phys))
        self.add_subprotocol(Distil(node_b, node_b.get_conn_port(node_a.ID, con_name), "B",
                                     name="purify_B",con_num = con_name, imp = inputmempos_b, phys = phys))
        
        self.add_subprotocol(DistilFurther(node_a, node_a.get_conn_port(node_b.ID, "AB1"), "A",
                                     name="purify_A1", con_num = "AB1", distilled_pos = 1, imp = inputmempos_a + 2, phys = phys))
        self.add_subprotocol(DistilFurther(node_b, node_b.get_conn_port(node_a.ID, "AB1"), "B",
                                     name="purify_B1",con_num = "AB1", distilled_pos = 1, imp = inputmempos_b + 2, phys = phys))
        self.add_subprotocol(DistilFurther(node_a, node_a.get_conn_port(node_b.ID, "AB2"), "A",
                                     name="purify_A2", con_num = "AB2", distilled_pos = 1, imp= inputmempos_a+3, phys = phys))
        self.add_subprotocol(DistilFurther(node_b, node_b.get_conn_port(node_a.ID, "AB2"), "B",
                                     name="purify_B2",con_num = "AB2", distilled_pos = 1, imp= inputmempos_b+3, phys = phys))

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
    conn_cchannel = DirectConnection(
        f"CChannelConn_{nodeAltr}{nodeBltr}",
        ClassicalChannel(f"CChannel_{nodeAltr}->{nodeBltr}_{con_num*2+1}", length=node_distance,
                         models={"delay_model": FibreDelayModel(c=200e3)}),
        ClassicalChannel(f"CChannel_{nodeBltr}->{nodeAltr}_{con_num*2+1}", length=node_distance,
                         models={"delay_model": FibreDelayModel(c=200e3)}))    
    network.add_connection(nodea, nodeb, connection=conn_cchannel, label = f"{nodeAltr}{nodeBltr}{con_num+1}")
    
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



def create_iters_plot(num_runs=10):
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    data = pandas.DataFrame()
    for i in range(num_runs):
        for num_iters in [1,2,3,4,5,10,15,20,100]:

            temp = simulate_two_nodes(node_distance=1, num_iters_dist=num_iters)
            data = pandas.concat([data,temp], ignore_index=True) 

        print(i)

    grouped = data.groupby('num_runs')['F2'].mean()
    print(grouped)
    plt.scatter(grouped.index, grouped.values)
    plt.title("Link fidelity wrt num iterations")
    plt.xlabel('Number of iterations')
    plt.ylabel("Fidelity")
    plt.show()


def create_dist_plot(num_iters=1, num_runs=1):

    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    data = pandas.DataFrame()
    for i in range(num_runs):
        for distance in [1,2,3,4,5,10,15,20]:
            temp = simulate_two_nodes(node_distance=distance, num_iters_dist=num_iters)
            data = pandas.concat([data,temp], ignore_index=True) 

        print(i)
                                    
    #     data[distance] = simulate_two_nodes(
    #                                         node_distance=distance,
    #                                         num_iters=num_iters)['F2']
    #     # For errorbars we use the standard error of the mean (sem)
    #     data = data.agg(['mean', 'sem']).T.rename(columns={'mean': 'F2'})
    #     data.plot(y='F2', yerr='sem', label=f"{distance} km", ax=ax)
    # plt.xlabel("Node dist")
    # plt.ylabel("Fidelity")
    # plt.title("Repeater chain with varying lengths")
    # plt.show()

    grouped = data.groupby('dist')['F2'].mean()
    print(grouped)
    plt.scatter(grouped.index, grouped.values)
    plt.title("Link fidelity wrt distance")
    plt.xlabel('ditance(km)')
    plt.ylabel("Fidelity")
    plt.show()

def bar_chart_(d1, d2, d3):
    fig = plt.figure(figsize = (10, 5))
 
    # creating the bar plot
    labels= ["1 round", "2 rounds", "3 rounds"]
    times = [d1['time'].mean(), d2['time'].mean(), d3['time'].mean()]/10*3
   
    plt.bar(labels, times, color ='maroon',
            width = 0.4)
    plt.ylabel("Time taken to distil (microseconds)")
    plt.xlabel("Number or links distilled")
    plt.title("Average time taken for different distillation schemes")
    plt.show()

def timetaken_plot(num_iters=500):
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    for dist_runs in [1,2,3]:
        data = pandas.DataFrame()
        for d in range(1, 30, 3):
            data[d] = simulate_two_nodes(sf = 0.8, node_distance=d, num_iters=num_iters, distruns = dist_runs)['time']
        # For errorbars we use the standard error of the mean (sem)
        data = data.agg(['mean', 'sem']).T.rename(columns={'mean': 'fidelity'})

        data.plot(y='fidelity', yerr='sem', label=f"{dist_runs} Distillation runs", ax=ax)
    plt.xlabel("distance (km))")

    plt.ylabel("Time taken to distil (s)")
    plt.title("Distillation perfect memories")
    plt.show()
    
# t1 = 3600*1e9,t2 = 1.46e9
def source_fidelity_plot(num_iters=10, t1 = 2.68e6, t2= 3.3e3, node_distance= 1, pln=0):
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    for dist_runs in [0,1,2,3]:
        if dist_runs == 0: 
            num_iters = 300 
        else :
            num_iters = 200
        data = pandas.DataFrame()
        for sf in range(8, 20):
            data[sf*.05] = simulate_two_nodes(p_loss_node=0, t1 = t1, t2 = t2, sf = sf*.05, node_distance=node_distance, num_iters=num_iters, distruns = dist_runs)['F2']
        # For errorbars we use the standard error of the mean (sem)
        data = data.agg(['mean', 'sem']).T.rename(columns={'mean': 'fidelity'})
        data.plot(y='fidelity', yerr='sem', label=f"m = {dist_runs+1}:1", ax=ax)
        print("done")
    plt.xlabel("Source Fidelity")
    plt.ylabel("Fidelity of link")
    plt.title(f"Single link, T1={t1},T2={t2}, Distance={node_distance}, Runs={num_iters}")
    plt.show()

def t1t2_plot(num_iters=100):
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(ncols=2)
    # (2.68e6, 3.3e3, 0), (2.68e6, 3.3e3, 1), (2.68e6, 3.3e3, 2), (2.68e6, 3.3e3, 3), (3600*1e9,1.46e9, 0), (3600*1e9,1.46e9, 1), (3600*1e9,1.46e9, 2), (3600*1e9,1.46e9, 3)
    # (2.68e6, 3.3e3, 0), (2.68e6, 3.3e3, 1), (2.68e6, 3.3e3, 2), (2.68e6, 3.3e3, 3)
    for t1 in [1e12, 1e11, 1e10, 1e9] :
        data = pandas.DataFrame()
        time_data = pandas.DataFrame()
        for t2 in [1e9, 5e8, 1e8, 5e7, 1e7, 5e6, 1e6]:
            res = simulate_two_nodes(t1 = t1, t2=t2, node_distance=1, num_iters=num_iters, distruns = 2)
            time_data[t2] = pow(res['time'], -1)*1e9
            data[t2] = res['F2']        

        data = data.agg(['mean', 'sem']).T.rename(columns={'mean': 'fidelity'})
        time_data = time_data.agg(['mean', 'sem']).T.rename(columns={'mean': 'time'})
        data.plot(y='fidelity', yerr='sem', label=f"T1={t1}, T2={t2}", ax=axes[0])
        time_data.plot(y='time', yerr='sem', label=f"T1={t1}, T2={t2}", ax=axes[1])
        print(f"done {t1}" )
  
    # plt.xlabel("T2 (ns)")
    # plt.ylabel("Avg Fidelity On success")
    # plt.title("Avg Fidelity for different T1, T2, 2:1, D=1km")
    # plt.show()
    # plt.title("Fidelity for different T1, T2, K-shortest paths")

    fig.suptitle("Fidelity for different T1, T2, K-shortest paths")
    axes[0].set_xlabel("T2 (ns)")
    axes[0].set_ylabel("Average fidelity on success")

    axes[0].set_xscale("log")
    axes[0].set_title("Avg fidelity")
    axes[1].set_xlabel("T2 (ns)")
    axes[1].set_ylabel("ebit rate (ebits/s)")

    axes[1].set_xscale("log")
    axes[1].set_title("Average ebit rate")
    plt.show()



def physical_plot(num_iters=200):
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(ncols=2)
 
    times = []
    # for nodes, distruns in (2,0), (2,1), (3,0), (3,1), (4,0), (4,1):
    for phys, distruns in  (True,1), (True,2), (True,3), (False,1), (False,2), (False,3):
        data = pandas.DataFrame()
        time_data = pandas.DataFrame()
        for d in [0.1, 0.2, 0.3, 0.4, 0.5] :
            
            # t1 = 2.68e6, t2= 3.3e3
            res = simulate_two_nodes(t1 = 2.68e6, t2= 3.3e3, node_distance=d, num_iters=num_iters, distruns = distruns, physical = phys)

            if not res.empty:
                if res.shape[0]!=num_iters:     #entanglesetup only returns value on success. 
                    # res['F2']=res['F2']*(res.shape[0]/num_iters) # we take a failure to entangle as 0 fidelity. This has the same affect as adding the failures as 0s then averaging.
                    res['time']=res['time']*(num_iters/res.shape[0])  # multiplying by the number of iters/number of outputs gives the total time
                time_data[d] = pow(res['time'], -1)*1e9
                data[d] = res['F2']
            
        print("done: {}".format(distruns))
        # For errorbars we use the standard error of the mean (sem)
        data = data.agg(['mean', 'sem']).T.rename(columns={'mean': 'fidelity'})
        time_data = time_data.agg(['mean', 'sem']).T.rename(columns={'mean': 'time'})
        
        data.plot(y='fidelity', yerr='sem', label=f"m = {distruns+1}:1, Phys = {phys}" , ax=axes[0])
        time_data.plot(y='time', yerr='sem', label=f"m = {distruns+1}:1, Phys = {phys}", ax=axes[1])
 
    fig.suptitle("Physical vs Non-physical Instructions - Electron Spins")
    axes[0].set_xlabel("Total Distance of link (km)")
    axes[0].set_ylabel("Average fidelity on success")
    axes[0].set_title("Avg fidelity")
    axes[1].set_xlabel("Total Distance of link (km)")
    axes[1].set_ylabel("ebit rate (ebits/s)")
    axes[1].set_title("Average ebit rate")

    plt.show()

def nodes_plot(num_iters=1):
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(ncols=2)
 
    times = []
    # for nodes, distruns in (2,0), (2,1), (3,0), (3,1), (4,0), (4,1):
    for nodes, distruns in  (2,0), (2,1), (2,2), (2,3):
        data = pandas.DataFrame()
        time_data = pandas.DataFrame()
        for d in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3] :
            
            res = simulate_two_nodes(t1 = 2.68e6, t2= 3.3e3, p_loss_node=(nodes)*2, node_distance=d, num_iters=num_iters, distruns = distruns)

            if not res.empty:
                if res.shape[0]!=num_iters:     #entanglesetup only returns value on success. 
                    # res['F2']=res['F2']*(res.shape[0]/num_iters) # we take a failure to entangle as 0 fidelity. This has the same affect as adding the failures as 0s then averaging.
                    res['time']=res['time']*(num_iters/res.shape[0])  # multiplying by the number of iters/number of outputs gives the total time
                time_data[d] = pow(res['time'], -1)*1e9
                data[d] = res['F2']


                print(d)
            
        print("done: {}".format(nodes))
        # For errorbars we use the standard error of the mean (sem)
        data = data.agg(['mean', 'sem']).T.rename(columns={'mean': 'fidelity'})
        time_data = time_data.agg(['mean', 'sem']).T.rename(columns={'mean': 'time'})
        
        data.plot(y='fidelity', yerr='sem', label=f"m = {distruns+1}:1, hops = {nodes-2}" , ax=axes[0])
        time_data.plot(y='time', yerr='sem', label=f"m = {distruns+1}:1, hops = {nodes-2}", ax=axes[1])
 
    # fig.suptitle("Dynamical Decoupling NV centres")
    fig.suptitle("Electron spins - NV centres")
    axes[0].set_xlabel("Total Distance of link (km)")
    axes[0].set_ylabel("Average fidelity on success")
    axes[0].set_title("Avg fidelity")
    axes[1].set_xlabel("Total Distance of link (km)")
    axes[1].set_ylabel("ebit rate (ebits/s)")
    axes[1].set_title("Average ebit rate")

    plt.show()


def simulate_two_nodes(t1= 3600 * 1e9, t2=1.46e9, node_distance=1, num_iters=1, sf=0.8, distruns =1, p_loss_node=2, physical = False):
    ns.sim_reset()
    if(distruns ==0):
        network = example_network_setup(t1 = t1, t2 =t2,  p_loss_node=p_loss_node, qprocessor_positions=4, node_distance=node_distance, source_fidelity_sq=sf, physical = physical)
        entangle = EntangleSetup(network.get_node("node_A"), network.get_node("node_B"), inputmempos_a=0, inputmempos_b= 0, num_runs=num_iters, con_name= "AB0", qs_no="0")
        dc = setup_data_collector(entangle, network.get_node("node_A"), network.get_node("node_B"))
        entangle.start()  

    elif(distruns ==1):
        network = example_network_setup(t1 = t1, t2 =t2,  p_loss_node=p_loss_node, qprocessor_positions=4, node_distance=node_distance, source_fidelity_sq=sf, physical = physical)
        dist_once = DistilSetup(network.get_node("node_A"), network.get_node("node_B"), inputmempos_a=0, inputmempos_b= 0, num_runs=num_iters, con_name= "AB0", qs_no="0", phys = physical)
        dc = setup_data_collector(dist_once, network.get_node("node_A"), network.get_node("node_B"))
        dist_once.start()  
        
    elif(distruns ==2):
        network = example_network_setup(t1 = t1, t2 =t2, qprocessor_positions=4, p_loss_node=p_loss_node, node_distance=node_distance, source_fidelity_sq=sf, physical = physical)
        add_connection(network, network.get_node("node_A"), network.get_node("node_B"), src_conns=1, dest_conns=1, con_num=1, node_distance=node_distance)
        distil_twice = DistilTwice(network.get_node("node_A"), network.get_node("node_B"), inputmempos_a=0, inputmempos_b= 0, num_runs=num_iters, con_name= "AB0", qs_no="0", phys = physical)
        dc = setup_data_collector(distil_twice, network.get_node("node_A"), network.get_node("node_B"))
        distil_twice.start()  

    elif(distruns ==3):
        network = example_network_setup(t1 = t1, t2 =t2, qprocessor_positions=6, p_loss_node=p_loss_node, node_distance=node_distance, source_fidelity_sq=sf, physical = physical)
        add_connection(network, network.get_node("node_A"), network.get_node("node_B"), src_conns=1, dest_conns=1, con_num=1, node_distance=node_distance)
        distil_thrice = DistilThrice(network.get_node("node_A"), network.get_node("node_B"), inputmempos_a=0, inputmempos_b= 0, num_runs=num_iters, con_name= "AB0", qs_no="0", phys = physical)
        dc = setup_data_collector(distil_thrice, network.get_node("node_A"), network.get_node("node_B"))
        distil_thrice.start()  

    ns.sim_run()

    return dc.dataframe


if __name__ == "__main__":
    
    # ns.sim_reset()

    # network = example_network_setup(qprocessor_positions=4, node_distance=1)
    # add_connection(network, network.get_node("node_A"), network.get_node("node_B"), src_conns=1, dest_conns=1, con_num=1, node_distance=1)
    # dist_once = DistilThrice(network.get_node("node_A"), network.get_node("node_B"), inputmempos_a=0, inputmempos_b= 0, num_runs=100, con_name= "AB0", qs_no="0")
    # dc = setup_data_collector(dist_once, network.get_node("node_A"), network.get_node("node_B"))
    # dist_once.start()  
    # ns.sim_run()
    np.random.seed(104058)
    # physical_plot()
    # source_fidelity_plot()
    t1t2_plot(5000)
