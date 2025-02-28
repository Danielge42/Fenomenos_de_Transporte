using NeuralPDE, Lux, Optimization, OptimizationOptimJL
import ModelingToolkit: Interval

@parameters r
@variables u(..)
Dr = Differential(r)
Drr= Differential(r)^2
L=2
mu=3
p0=10
pf=3
p=(p0-pf)
R=2
# 2D PDE
eq = r*Drr(u(r)) + Dr(u(r)) ~ -p*r/(mu*L)

# Boundary conditions
bcs = [u(R) ~ 0.0, u(-R) ~ -0.0]
# Space and time domains
domains = [r ∈ Interval(-R, R)]

# Neural network
dim = 1 # number of dimensions
chain = Lux.Chain(Lux.Dense(dim, 4, Lux.σ), Lux.Dense(4, 4, Lux.σ), Lux.Dense(4, 1))
i
# Discretization
dr = 0.05
discretization = PhysicsInformedNN(chain, GridTraining(dr))

@named pde_system = PDESystem(eq, bcs, domains, [r], [u(r)])    
prob = discretize(pde_system, discretization)

#Optimizer
opt = OptimizationOptimJL.BFGS()

#Callback function
iter=0
callback = function (p, l)
    global iter += 1
    if iter % 100 == 0
        println("Current loss is: $l")
    end
    return false
end

res = Optimization.solve(prob, opt, callback = callback, maxiters = 5000)
phi = discretization.phi

using Plots

rsr = [infimum(d.domain):(dr):supremum(d.domain) for d in domains][1]
rsp= [infimum(d.domain):(dr):supremum(d.domain) for d in domains][1]
u_real= p/(4*mu*L)*R^2*(1-(r/R)^2)
typeof(u_real)
u_predict = [first(phi(r, res.u)) for r in rsp]
plot(rsp,u_predict,label="predict", ls=:dot)
plot!(rsr,u_real,label="real")
#p1 = plot(rs, u_predict, linetype = :contourf, title = "Primer Intento");
#plot(p1)

