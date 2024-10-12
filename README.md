# DDP_iLQR_Variants

Library serves as a codespace for all my code for the master thesis: 

**Efficient Ineqaulity Constrained Model Predictive Control on Matrix Lie Group**.

## File Structure

The folder `traoptlibrary` is the main library developed independently. It contains:

- `traopt_controller.py`
- `traopt_cost.py`
- `traopt_dynamics.py`
- `traopt_utilis.py`

The main files in the root directory define tasks that invoke the appropriate modules from the `traoptlibrary`.

- `main_ddp.py`: cartpole system swingup task, general task, nominal ddp in Euclidean space.
- `main_SE3ddp_tracking.py`: SE3 reference trajectory tracking task, implemented both with exact nonlinear dynamics and approximate error-state dynamics.
- `main_errSE3ddp_linear_rollout_generation.py`: SE3 trajectory generation task, with approximate error-state(vector) dynamics, using linear rollout.
- `main_errSE3ddp_nonlinear_rollout_generation.py`: SE3 trajectory generation task, with approximate vector error-state dynamics, using nonlinear rollout.
- `main_SE3dynamics.py`: SE3 nonlinear dynamics simulation, state described by quaternion-position-stacked vector.
- `main_errSE3dynamics.py`: SE3 approximate error-state dynamics simulation, to compare the approximate vector error-state dynamics and the exact nonlinear dynamics when rolling-out.
