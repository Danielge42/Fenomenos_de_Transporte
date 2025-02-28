a = 1
using Pkg
Pkg.add("LineSearches")
Pkg.add("ModelingToolkit")
Pkg.add("Optimization")
Pkg.add("OptimizationOptimJL")
Pkg.add("NeuralPDE")
Pkg.add("Lux")
Pkg.add("Plots")
using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL, LineSearches

using ModelingToolkit: Interval, infimum, supremum


@parameters x, y
@variables u(..), v(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2
Dx = Differential(x)
Dy = Differential(y)
mu = 1
rho = 1
dpx = -10
dpy = 0
lx = 10
ly = 1


eq = [mu/rho * (Dxx(u(x,y)) + Dyy(u(x,y))) - 1/rho *dpx - u(x,y)*Dx(u(x,y)) - v(x,y)*Dy(u(x,y))~0,
        mu/rho * (Dxx(v(x,y)) + Dyy(v(x,y))) - 1/rho *dpy - u(x,y)*Dx(v(x,y)) - v(x,y)*Dy(v(x,y))~0,
        Dx(u(x,y)) + Dy(v(x,y))~0]

bcs = [ u(x,0)~0, v(0,y)~0, v(x,1)~0, u(x,1)~0]

domains = [x ∈ Interval(0,1), y  ∈ Interval(0,1)]

@named pde_system = PDESystem(eq,bcs,domains,[x,y],[u,v])



input_ = length(domains)
n = 60
chain = [Lux.Chain(Lux.Dense(input_, n, Lux.σ), Lux.Dense(n, n, Lux.σ), Lux.Dense(n, 1)) for _ in 1:2]

chain

strategy =  QuadratureTraining(; batch = 200, abstol = 1e-6, reltol = 1e-6)

discretization = PhysicsInformedNN(chain, strategy)    

discretization

prob = discretize(pde_system , discretization)

prob = discretize(pde_system, discretization)
sym_prob = symbolic_discretize(pde_system, discretization)


pde_inner_loss_functions = sym_prob.loss_functions.pde_loss_functions
bcs_inner_loss_functions = sym_prob.loss_functions.bc_loss_functions



callback = function (p,l)
    println("loss: ", l)
    println("pde_losses: ", map(l_ -> l_(p.u), pde_inner_loss_functions))
    println("bcs_losses: ", map(l_ -> l_(p.u), bcs_inner_loss_functions))
    return false
end


res = solve(prob, LBFGS(linesearch = BackTracking()); maxiters = 1000)
phi = discretization.phi

using Pkg
Pkg.gc()
Pkg.rm("Plots")
Pkg.add("Plots")
using Plots

xs, ys = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]
minimizers_ = [res.u.depvar[sym_prob.depvars[i]] for i in 1:2]
u_predict = [[phi[i]([x, y], minimizers_[i])[1] for x in xs for y in ys] for i in 1:2]

plot(xs, ys, reshape(u_predict[1], length(xs), length(ys)), linetype = :contourf, title = "Vx", xlabel = "x[m]",ylabel = "y[m]")


plot(xs, ys, reshape(u_predict[2], length(xs), length(ys)), linetype = :contourf, title = "u2, predict")
u_x = u_predict[1]

xq =xs# range(0,stop=1,length=2)
yq = ys#range(0,stop=1,length=100)

quiver(xq, yq,quiver= (u_predict[1],u_predict[2]), title = " Campo de velocidades", xlabel = "Posición[m]",ylabel="Velocidad[m/s]" )
u_real = [ dpx/(2*mu)*y*(y-ly) for y in yq, x in xq]
v_real = [0*x*y for  y in yq,x in xq]
quiver!(xq,yq, quiver=(u_real,v_real), title = "Campo de velocidades", xlabel = "Posición[m]",ylabel="Velocidad[m/s]", label = "Velocidad real", line = :dash)
u_predict[1]
vec(u_real)
using Statistics
mean(abs.(vec(u_real) - u_predict[1]))*100/mean(abs.(u_real))