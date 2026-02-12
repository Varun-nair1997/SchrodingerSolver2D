__author__ = 'Varun Nair'

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as lg


class Schrodinger2D:
    """
    Finite-difference solver for the time-independent 2D Schrödinger equation
    on a periodic square domain.
    The class builds a uniform 2D grid and represents the Hamiltonian operator
    using a second-order finite-difference Laplacian with periodic boundary
    conditions (implemented via np.roll). Eigenstates are computed using an
    iterative residual minimization (imaginary-time–like) scheme with optional
    Gram–Schmidt orthogonalization to previously computed states.
        Attributes
        ----------
        x, y : ndarray
            1D spatial grids.
        xx, yy : ndarray
            2D meshgrid arrays.
        dx, dy : float
            Grid spacings.
        waves : list
            List of converged eigenstates.
        period : float
            Spatial period of the potential.

    """
    def __init__(self, mesh=100, x_length=4 * np.pi, y_length=4 * np.pi, pot_type='cosine', v_depth_const=5, n_atoms=1):
        """
        Initialize the 2D grid, potential parameters, and solver state.
        Constructs uniform spatial grids in x and y, computes mesh spacing,
        builds 2D meshgrids, and stores parameters defining the potential
        and periodicity.
        :param mesh: int
            Number of grid points per spatial dimension.
        :param x_length: float
            Physical length of the domain in the x direction.
        :param y_length: float
            Physical length of the domain in the y direction.
        :param pot_type: str
            Type of potential to generate ('cosine', 'sine', or 'well').
        :param v_depth_const: float
            Depth of the potential well (used for 'well' type).
        :param n_atoms: int
            Number of periodic cells along each direction.
        """
        self.mesh = mesh
        self.n_atoms = n_atoms
        self.x_length = x_length
        self.y_length = y_length
        self.x, self.dx = np.linspace(0, x_length, mesh, endpoint=False, retstep=True)
        self.y, self.dy = np.linspace(0, y_length, mesh, endpoint=False, retstep=True)
        self.xx, self.yy = np.meshgrid(self.x, self.y)
        self.pot_type = pot_type
        self.v_depth_const = v_depth_const
        self.waves = []
        self.period = self.x_length / self.n_atoms

    # ---------- Utilities ----------

    def get_2D_potential(self):
        """
        Return the spatial potential evaluated on the 2D grid.
        Depending on `pot_type`, generates a periodic trigonometric potential
        or a rectangular well. The result is a (mesh × mesh) array representing
        V(x, y).
        :return: pot: ndarray
            matrix representation of potential
        """
        if self.pot_type == 'cosine':
            return -10 * ((np.cos(self.xx * (2 * np.pi / self.period)) - 1) / 2 * (
                        np.cos(self.yy * (2 * np.pi / self.period)) - 1) / 2)

        elif self.pot_type == 'sine':
            return -10 * ((np.sin(self.xx * (2 * np.pi / self.period)) - 1) / 2 * (
                        np.sin(self.yy * (2 * np.pi / self.period)) - 1) / 2)

        elif self.pot_type == 'fancy1':
            return -10 * ((np.sin(self.xx * (2 * np.pi / self.period)) - 1) / 2 * (
                        np.cos(self.yy * (2 * np.pi / self.period)) - 1) / 2)

        elif self.pot_type == 'fancy2':
            return -10 * ((np.cos(self.xx * (2 * np.pi / self.period)) - 1) / 2 * (
                        np.sin(self.yy * (2 * np.pi / self.period)) - 1) / 2)

        elif self.pot_type == 'well':
            pot = np.zeros((self.mesh, self.mesh))
            mask = ((self.xx > 0.25 * self.xx.max()) & (self.xx < 0.75 * self.xx.max()) &
                    (self.yy > 0.25 * self.yy.max()) & (self.yy < 0.75 * self.yy.max()))
            pot[mask] = -self.v_depth_const
            return pot

        else:
            print("WARNING: unknown potential, self destruct in 5... 4... 3... 2...... Bye-bye, kaboom!!!")
            return None  # np.sin(self.x)

    def random_psi(self):
        """
        Generate a random normalized wavefunction on the grid.
        :return: psi: ndarray
            Random real-valued wavefunction normalized to unit L2 norm.
        """
        psi = np.random.random_sample(self.xx.shape)
        return self.normalize(psi)

    def normalize(self, psi):
        """
        Normalize a wavefunction to unit L2 norm.
        :param psi:  ndarray
            Wavefunction to normalize.
        :return: ndarray
            |input| = 1
        """
        return psi / np.linalg.norm(psi)

    # ---------- Operators ----------

    def laplacian_2D(self, psi):
        """
        Compute the discrete 2D Laplacian using second-order finite differences.
        Periodic boundary conditions are enforced using array rolling.
        :param psi: ndarray
            Wavefunction on the grid.

        :return: ndarray
            Discrete Laplacian of psi.
        """
        lap_x = (
                        np.roll(psi, -1, axis=0)
                        - 2 * psi
                        + np.roll(psi, 1, axis=0)
                ) / self.dx ** 2

        lap_y = (
                        np.roll(psi, -1, axis=1)
                        - 2 * psi
                        + np.roll(psi, 1, axis=1)
                ) / self.dy ** 2

        return lap_x + lap_y

    def hamiltonian(self, psi):
        """
        Apply the Hamiltonian operator to a wavefunction.
        The Hamiltonian is defined as:
            H = [-1/2 ∇² + V(x, y)]
        :param psi: psi : ndarray
            Wavefunction on the grid.
        :return: ndarray
            'hampsi' hamiltonian applied to psi
        """
        kinetic = -0.5 * self.laplacian_2D(psi)
        potential = self.get_2D_potential() * psi
        return kinetic + potential

    # ---------- Physics ----------

    def energy(self, psi):
        """
        Compute the expectation value of the Hamiltonian.
        :param psi: ndarray
            Wavefunction on the grid (assumed normalized).
        :return: float
            Energy expectation value <psi|H|psi>.
        """
        Hpsi = self.hamiltonian(psi)
        return np.sum(psi * Hpsi)

    def residual(self, psi):
        """
        Compute the eigenvalue residual for a wavefunction.
        The residual is defined as:
            r = Hψ − Eψ
        where E is the energy expectation value.

        :param psi: ndarray
            Wavefunction on the grid.
        :return: ndarray
            Residual vector measuring deviation from an eigenstate.
        """
        E = self.energy(psi)
        Hpsi = self.hamiltonian(psi)
        return Hpsi - E * psi

    def next_psi(self, psi, step=-0.01):
        """
        Perform one iterative update step toward an eigenstate.
        Updates the wavefunction along the residual direction and
        renormalizes it.
        :param psi: ndarray
            Current wavefunction.
        :param step: float
            Step size controlling convergence rate.
        :return: ndarray
            Updated and normalized wavefunction.
        """
        r = self.residual(psi)
        psi_next = psi + step * r
        return self.normalize(psi_next)

    # ---------- Gram-Schmidt Process ---------

    def project_out(self, psi, states):
        """
        Project a wavefunction onto the subspace orthogonal to given states.
        :param psi:ndarray
            Wavefunction to be orthogonalized.
        :param states: list of ndarray
            Previously computed orthonormal states.
        :return: ndarray
            Wavefunction with components along `states` removed.
        """
        psi_ortho = psi.copy()
        for phi in states:
            psi_ortho -= np.sum(phi * psi_ortho) * phi
        return psi_ortho

    def orthonormalize(self, psi, states):
        """
        Orthogonalize and normalize a wavefunction against given states.

        Applies Gram–Schmidt projection followed by normalization.
        :param psi: ndarray
            Wavefunction to orthonormalize.
        :param states: list of ndarray
            States to orthogonalize against.
        :return: ndarray
            Orthonormalized wavefunction.
        """
        psi = self.project_out(psi, states)
        return self.normalize(psi)

    def solve(self, n_steps=2500, step=-0.01, tol=1e-3,
              psi0=None, orthogonal_to=None, store_history=True):
        """
        Iteratively solve for an eigenstate of the Hamiltonian.
        Uses residual minimization with optional orthogonalization to
        previously computed states to obtain excited states.
        :param n_steps: int
            Maximum number of iterations.
        :param step: float
            learning rate hyperparameter (like in ML)
        :param tol: float
            Convergence tolerance based on residual norm.
        :param psi0: ndarray or None
            Optional initial wavefunction guess.
        :param orthogonal_to: list of ndarray or None
            States to orthogonalize against (for excited states).
        :param store_history:bool
            If True, stores intermediate wavefunctions.

        :return:
        psi : ndarray
            Converged eigenstate.
        E : float
            Corresponding energy eigenvalue.
        """

        if psi0 is None:
            psi = self.random_psi()
        else:
            psi = self.normalize(psi0)

        if orthogonal_to is None:
            orthogonal_to = []

        # Enforce orthogonality at start
        if orthogonal_to:
            psi = self.orthonormalize(psi, orthogonal_to)

        self.psis = [psi]
        self.energies = []

        for i in range(n_steps):
            E = self.energy(psi)
            r = self.residual(psi)
            r_norm = np.linalg.norm(r)

            self.energies.append(E)

            if r_norm < tol:
                print(f"Converged at step {i}, residual = {r_norm:.2e}")
                break

            psi = self.next_psi(psi, step=step)

            # Gram–Schmidt every iteration
            if orthogonal_to:
                psi = self.orthonormalize(psi, orthogonal_to)

            if store_history:
                self.psis.append(psi)

        self.waves.append(psi)
        return psi, E

    def solve_n_states(self, n_states=4, n_steps=2500, step=-0.01, tol=1e-3):
        """
        Compute multiple eigenstates using Gram–Schmidt orthogonalization.

        :param n_states: int
            Number of eigenstates to compute (including ground state).
        :param n_steps: int
            Maximum iterations per state.
        :param step: float
            Iteration step size.
        :param tol: float
            Convergence tolerance.
        :return:
            states : list of ndarray
            energies : list of float
        """
        states = []
        energies = []

        for n in range(n_states):
            print(f"Solving state {n}...")
            psi, E = self.solve(
                n_steps=n_steps,
                step=step,
                tol=tol,
                orthogonal_to=states if states else None
            )
            states.append(psi)
            energies.append(E)

        return states, energies

    def plot_states(self, states, energies):
        """
        Plot computed eigenstates on the 2D spatial grid.
        Generates a row of subplots displaying each wavefunction using
        imshow, along with its corresponding energy eigenvalue in the title.
        The spatial extent matches the physical domain defined by
        `x_length` and `y_length`.

        :param states: list of ndarray
            List of eigenstates to plot. Each state must be a
            (mesh × mesh) array defined on the solver grid.
        :param energies: list of float
            Energy eigenvalues corresponding to each state.
        :return:
            None
        """
        n = len(states)
        fig, axes = plt.subplots(1, n, figsize=(4*n, 4))

        if n == 1:
            axes = [axes]

        for i, (psi, E) in enumerate(zip(states, energies)):
            ax = axes[i]
            im = ax.imshow(psi, extent=[0, self.x_length, 0, self.y_length])
            ax.set_title(f"State {i}\nE = {E:.4f}")
            plt.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    solver = Schrodinger2D(
        mesh=50,
        x_length=4*np.pi,
        y_length=4*np.pi,
        pot_type='cosine',
        n_atoms=1
    )

    n_states = 4

    states, energies = solver.solve_n_states(
        n_states=n_states,
        n_steps=2500,
        step=-0.01,
        tol=1e-3
    )

    solver.plot_states(states, energies)
