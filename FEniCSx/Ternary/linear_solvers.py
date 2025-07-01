import ufl
from dolfinx import fem, io, common
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx.fem.petsc

import numpy as np
from typing import List, Union, Dict, Optional, Callable
import time
import logging
LOG_INFO_STAR = logging.INFO + 5
LOG_INFO_TIME_ONLY = LOG_INFO_STAR + 1

SQRT2 = np.sqrt(2.)

class LinearProblem():
    def __init__(
        self,
        dR: ufl.Form,
        R: ufl.Form,
        u: fem.Function,
        bcs: List[fem.dirichletbc] = []
    ):
        self.u = u
        self.bcs = bcs

        V = u.function_space
        domain = V.mesh

        self.R = R
        self.dR = dR
        self.b_form = fem.form(R)
        self.A_form = fem.form(dR)
        self.b = dolfinx.fem.petsc.create_vector(self.b_form)
        self.A = dolfinx.fem.petsc.create_matrix(self.A_form)

        self.comm = domain.comm

        self.solver = self.solver_setup()

    def solver_setup(self) -> PETSc.KSP:
        """Sets the solver parameters."""
        solver = PETSc.KSP().create(self.comm)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setOperators(self.A)
        return solver

    def assemble_vector(self) -> None:
        with self.b.localForm() as b_local:
            b_local.set(0.0)
        fem.petsc.assemble_vector(self.b, self.b_form)
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(self.b, self.bcs)

    def assemble_matrix(self) -> None:
        self.A.zeroEntries()
        fem.petsc.assemble_matrix(self.A, self.A_form, bcs=self.bcs)
        self.A.assemble()

    def assemble(self) -> None:
        self.assemble_matrix()
        self.assemble_vector()
    
    def solve (
        self, 
        du: fem.function.Function, 
    ) -> None:
        """Solves the linear system and saves the solution into the vector `du`
        
        Args:
            du: A global vector to be used as a container for the solution of the linear system
        """
        self.solver.solve(self.b, du.x.petsc_vec)

class NonlinearProblem(LinearProblem):
    def __init__(
        self,
        dR: ufl.Form,
        R: ufl.Form,
        u: fem.Function,
        bcs: List[fem.dirichletbc] = [],
        Nitermax: int = 200, 
        tol: float = 1e-8,
        inside_Newton: Optional[Callable] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(dR, R, u, bcs)
        self.Nitermax = Nitermax
        self.tol = tol
        self.du = fem.Function(self.u.function_space)

        if inside_Newton is not None:
            self.inside_Newton = inside_Newton
        else:
            def dummy_func():
                pass
            self.inside_Newton = dummy_func
        
        if logger is not None:
            self.logger = logger 
        else:
            self.logger = logging.getLogger('nonlinear_solver')

    
    def solve(self) -> int:
        
        self.assemble()

        nRes0 = self.b.norm() # Which one? - ufl.sqrt(Res.dot(Res))
        nRes = nRes0
        niter = 0

        # start = time.time()

        while nRes/nRes0 > self.tol and niter < self.Nitermax:
            
            self.solver.solve(self.b, self.du.vector)
            
            self.u.vector.axpy(1, self.du.vector) # u = u + 1*du
            self.u.x.scatter_forward() 

            start_return_mapping = time.time()

            self.inside_Newton()

            end_return_mapping = time.time()

            self.assemble()

            nRes = self.b.norm()

            niter += 1

            self.logger.debug(f'rank#{MPI.COMM_WORLD.rank}  Increment: {niter}, norm(Res/Res0) = {nRes/nRes0:.1e}. Time (return mapping) = {end_return_mapping - start_return_mapping:.2f} (s)')
            
        
        # self.logger.debug(f'rank#{MPI.COMM_WORLD.rank}  Time (Step) = {time.time() - start:.2f} (s)')
        # self.logger.log(LOG_INFO_STAR, f'rank#{MPI.COMM_WORLD.rank}: Step: {str(i+1)}, Iterations = {niter}, Time = {time.time() - start:.2f} (s) \n')


        return niter