using NonlinearSolve
using Krylov


function example_residual!(F, u, p)
    @. F = u*u - 1
end

c_guess = rand(10)         
p = (dummy = 1.0,)        
problem = NonlinearProblem(example_residual!, c_guess, p)

solver = NewtonRaphson(linsolve = KrylovJL_GMRES())


sol = solve(problem, solver,termination_condition = AbsNormSafeTerminationMode(NonlinearSolve.L2_NORM))

println("Final solution: ", sol.u)
println("Return code: ", sol.retcode)