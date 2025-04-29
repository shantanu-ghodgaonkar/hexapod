import numpy as np
import cyipopt
from hexapod_v2_5_5 import hexapod
from time import time
from scipy.sparse import csr_matrix

import faulthandler
import signal
faulthandler.enable(all_threads=True)
# faulthandler.register(signal.SIGSEGV, all_threads=True, chain=False)


class MPCProblem:
    def __init__(self, hexapod, desired_seq, horizon):
        self.hexy = hexapod
        self.desired_seq = desired_seq
        self.horizon = horizon
        self.n_q = hexapod.robot.nq            # 25
        self.n_vars = self.n_q * horizon           # e.g. 75
        self.n_constr = 5 * horizon                  # 5 constraints per step

        # initial guess
        self.x0 = np.tile(self.hexy.qc, horizon)

        # variable bounds: list of (low,high) for each var
        lb = np.array([b[0] for b in self.hexy.bounds] * horizon, dtype=float)
        ub = np.array([b[1] for b in self.hexy.bounds] * horizon, dtype=float)
        self.lb = lb
        self.ub = ub

        # constraint bounds: equality constraints => cl=cu=0
        self.cl = np.zeros(self.n_constr)
        self.cu = np.zeros(self.n_constr)

        # Precompute sparsity of the linear constraint jacobian
        # 1) get the dense (15×75) array
        A_dense = self.hexy.mpc_eq_jacobian(self.x0, horizon)

        # 2) convert to CSR then to COO so you have a consistent (row,col,value) triplet
        A = csr_matrix(A_dense)      # now .data, .indices, .indptr define CSR
        coo = A.tocoo()              # coo.row, coo.col, coo.data all aligned

        # 3) stash them as Python lists
        self.jac_rows = coo.row.tolist()     # e.g. [0,0,0, ...]
        self.jac_cols = coo.col.tolist()     # e.g. [2,5,10,...]
        self.jac_vals_init = coo.data.tolist()    # the actual nonzero values

        # Precompute Hessian sparsity (Gauss–Newton blocks)
        # after you build H0 = self.hexy.mpc_hessian(self.x0, zeros)
        H0 = self.hexy.mpc_hessian(self.x0, np.zeros(self.n_constr)).tocoo()
        mask = H0.row >= H0.col              # keep only lower-triangular entries
        self.hess_rows = H0.row[mask].tolist()
        self.hess_cols = H0.col[mask].tolist()

    def objective(self, x):
        # scalar cost
        return float(self.hexy.mpc_cost(x, self.desired_seq, self.horizon))

    def gradient(self, x):
        # 1-D array length n_vars
        return self.hexy.mpc_cost_grad(x, self.desired_seq, self.horizon)

    def constraints(self, x):
        # 1-D array length n_constr
        return self.hexy.mpc_eq_constraints(x, self.horizon)

    def jacobian(self, x):
        # For linear constraints the jacobian is constant,
        # so we just return the precomputed values:
        return self.jac_vals_init

    def jacobianstructure(self):
        # row/col indices of nonzeros in the jacobian
        return (self.jac_rows, self.jac_cols)

    def hessianstructure(self):
        # IPOPT wants lists of row indices and column indices (both length = nnz)
        return (self.hess_rows, self.hess_cols)

    def hessian(self, x, lagrange, obj_factor):
        # rebuild the full GN Hessian in COO form
        Hfull = self.hexy.mpc_hessian(x, lagrange).tocoo()
        # pick out only the lower-triangle entries, same mask logic:
        mask = Hfull.row >= Hfull.col
        # scale by obj_factor (IPOPT’s τ in Lagrangian Hessian)
        vals = obj_factor * Hfull.data[mask]
        return vals.tolist()


# --- set up and solve with IPOPT ---


def solve_mpc_with_ipopt(hexapod, desired_seq, horizon):
    prob = MPCProblem(hexapod, desired_seq, horizon)

    nlp = cyipopt.Problem(
        n=prob.n_vars,
        m=prob.n_constr,
        problem_obj=prob,
        lb=prob.lb,
        ub=prob.ub,
        cl=prob.cl,
        cu=prob.cu
    )
    # options
    nlp.add_option('tol',          1e-6)
    nlp.add_option('max_iter',     200)
    nlp.add_option('print_level',  1)
    # nlp.add_option('hessian_approximation', 'limited-memory')

    # solve
    x_opt, info = nlp.solve(prob.x0)
    # reshape into (horizon × n_q) and return first step
    q_seq_opt = x_opt.reshape(horizon, prob.n_q)
    return q_seq_opt[0], info

# --- usage example ---
# qi, info = solve_mpc_with_ipopt(hexy, desired_seq=window, horizon=3)
# print("first-step q:", qi)
# print("IPOPT info:", info)


if __name__ == "__main__":

    # Create a hexapod instance with visualization and debug logging
    hexy = hexapod(init_viz=False, logging_level=50)

    # q_old = np.load('gait_angles/gait_angles_DIR_N_WP5_S1_20250406_134944.npy')
    # states = np.array([hexy.forward_kinematics(q_i) for q_i in q_old])
    # hexy.plot_trajctory(state=states, title='q old')
    # exit()
    # sleep(3)
    # Set parameters for movement
    v = 0.5  # Velocity in m/s
    # start_time = time()
    WAYPOINTS = 20
    # DIR = 'N'
    # start = time()
    q = np.copy(hexy.qc)

    horizon = 3

    wp = hexy.generate_waypoints(
        WAYPOINTS=WAYPOINTS, step_size_xy_mult=1, leg_set=0)
    wp = [wp[:, i].reshape(-1, 1) for i in range(wp.shape[1])]
    for i in range(len(wp)):
        window = wp[i:i + horizon]
        if len(window) < horizon:
            # pad with last element
            window += [wp[-1]] * (horizon - len(window))
        start = time()
        qi, info = solve_mpc_with_ipopt(hexy, desired_seq=window, horizon=3)
        print(f'Optimized in {time()-start}s')
        hexy.update_current_pose(q=qi)
        q = np.vstack((q, qi))

    wp = hexy.generate_waypoints(
        WAYPOINTS=WAYPOINTS, step_size_xy_mult=2, leg_set=1)
    wp = [wp[:, i].reshape(-1, 1) for i in range(wp.shape[1])]
    for i in range(len(wp)):
        window = wp[i:i + horizon]
        if len(window) < horizon:
            # pad with last element
            window += [wp[-1]] * (horizon - len(window))
        start = time()
        qi, info = solve_mpc_with_ipopt(hexy, desired_seq=window, horizon=3)
        print(f'Optimized in {time()-start}s')
        hexy.update_current_pose(q=qi)
        q = np.vstack((q, qi))

    wp = hexy.generate_waypoints(
        WAYPOINTS=WAYPOINTS, step_size_xy_mult=2, leg_set=0)
    wp = [wp[:, i].reshape(-1, 1) for i in range(wp.shape[1])]
    for i in range(len(wp)):
        window = wp[i:i + horizon]
        if len(window) < horizon:
            # pad with last element
            window += [wp[-1]] * (horizon - len(window))
        start = time()
        qi, info = solve_mpc_with_ipopt(hexy, desired_seq=window, horizon=3)
        print(f'Optimized in {time()-start}s')
        hexy.update_current_pose(q=qi)
        q = np.vstack((q, qi))

    wp = hexy.generate_waypoints(
        WAYPOINTS=WAYPOINTS, step_size_xy_mult=1, leg_set=1)
    wp = [wp[:, i].reshape(-1, 1) for i in range(wp.shape[1])]
    for i in range(len(wp)):
        window = wp[i:i + horizon]
        if len(window) < horizon:
            # pad with last element
            window += [wp[-1]] * (horizon - len(window))
        start = time()
        qi, info = solve_mpc_with_ipopt(hexy, desired_seq=window, horizon=3)
        print(f'Optimized in {time()-start}s')
        hexy.update_current_pose(q=qi)
        q = np.vstack((q, qi))
    q = np.delete(q, 0, axis=0)
    states = np.array([hexy.forward_kinematics(q_i) for q_i in q])
    hexy.plot_trajctory(state=states, title='v2.5.5 MPC')
    # hexy.viz.play(q)

    # gait_angles_file_path = Path(
    #     f'gait_angles/gait_angles_MPC_1.npy')
    # gait_angles_file_path.parent.mkdir(parents=True, exist_ok=True)
    # np.save(gait_angles_file_path, q)
