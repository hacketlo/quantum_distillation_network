# quantum_distillation_network

Quantum network simulator using netSquid.

To run these files you must:

1. install netsquid on a linux environment
2. obtain a free license by creating an account at netsquid.org
3. create a python environment using python 3.9 or less
4. pip install netsquid.

it is explained better on their website

--------------------------------------------------------------------


Three seperate files for three different networks, 
2 nodes and 6 nodes are mostly the same, except the network setup is slightlydifferent, as the two node case only needs two nodes.

WS network has some slight modifications to the route finding, but is otherwise vastly the same as the six_node file.

All the functions towards the bottom of each file are just utility functions for plotting. Calling any of these for 2 nodes will reproduce figures from the thesis. Six_node and WS_network figures were normally produced by running the runandsave function for given parameters, and then producing the graph using one of the utility functions that reads from the csv. This was done as these simulations can take 2+ hours.
