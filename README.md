# Nested-forward-diff-opt

An example of nested autodiff in an optimization problem.

For a system of ODE, it was assumed that the steady state value of one of the state variables was known, along with one of the parameters.  It was then assumed that the system was perturbed, with a decrease in production rate of the native form by inclusion of a labeled amino acid.  It was assumed that the fraction of labeled species was measured in the final compartment over 36 time units.  

The production rate constant kf is linearly related to steady state value of known state variable.  Thus, kf can be readily determined by optimization, solving the ODE for the steady state to find the value of kf that yields the known value of the state variable.  This can be wrapped within an optimization to determine the two unknown parameters (in p) from the kinetic data.

The optimization converges on the true values (but accuracy is sensitive to the solver).  The question remains: are the derivatives calculated correctly?  The finite difference gradient is similar to the forward diff gradient (through the loss function), but not exactly the same.
