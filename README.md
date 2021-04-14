# FlexPCB-Load-Cell-Model

This repo exists for fitting a Kalman filter to a set of measurement/state data.

In this case, the measurement is the capacitance of two plates in a physical device which are subject to some input force (one of the states).

The state vector consists of a number of internal displacement states and the input force. 

The model is assumed to be linear and the measurement is assumed to be the distance between two intermediate nodes (not necessarily consecutive)
