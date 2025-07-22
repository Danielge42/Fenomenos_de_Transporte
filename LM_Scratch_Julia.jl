using LinearAlgebra

using ForwardDiff

using Random
using LinearAlgebra
using Plots
using LeastSquaresOptim


# Constants
l0 = 10.0
w = Diagonal(ones(4))
a = 0.0
pro = 1e-16


# Define the system of equations
function equation_system(x)
    eq1 = - (2x[1] + 4x[1]*x[2] + 2x[2] + x[3])
    eq2 = - (4x[1] + 2x[2] - 3x[1]*x[3])
    eq3 = 1 - (x[1]*x[3] + 2x[2])
    eq4 = 2 - (1/x[1] + x[4] + x[4]^2)
    return [eq1, eq2, eq3, eq4]
end

# Define the output vector y(x)
function zzz(x)
    y1 = 2x[1] + 4x[1]*x[2] + 2x[2] + x[3]
    y2 = 4x[1] + 2x[2] - 3x[1]*x[3]
    y3 = x[1]*x[3] + 2x[2]
    y4 = 1/x[1] + x[4] + x[4]^2
    return [y1, y2, y3, y4]
end

# Jacobian using ForwardDiff
J_func = x -> ForwardDiff.jacobian(equation_system, x)

# h step calculation
function h(J, w, cost, x, l0)
    A = J' * w * J + l0 * Diagonal(diag(J' * w * J))
    return pinv(A) * J' * w * cost(x)
end

# ρ computation
function rho(J, x, h, w, cost)
    num = cost(x)' * w * cost(x) - cost(x + h)' * w * cost(x + h)
    den = abs(h' * (l0 * w * h + J' * w * cost(x)))
    return num / den
end

# Levenberg–Marquardt algorithm
function levenberg_marquardt(x0, w, cost, zzz; l0=10.0, max_iter=10000)
    x = copy(x0)
    for i in 1:max_iter
        J = ForwardDiff.jacobian(zzz, x)
        hlm_val = h(J, w, cost, x, l0)
        rho_val = rho(J, x, hlm_val, w, cost)
        println("ρ = $rho_val, h = $hlm_val, λ = $l0")
        if rho_val > 1e-7
            x += hlm_val
            l0 = max(l0 / 10, 1e-6)
        else
            l0 = min(l0 * 10, 1e7)
        end
        if norm(hlm_val) < eps()
            break
        end
    end
    return x
end

function rho_nonsquare(J, x, h, w,win, cost)
    num = cost(x)' * w * cost(x) - cost(x + h)' * w * cost(x + h)
    den = abs(h' * (l0 * win * h + J' * w * cost(x)))
    return num / den
end


function levenberg_marquardt_nonsquare(x0, w, cost, zzz,win; l0=10.0, max_iter=50)
    x = copy(x0)
    for i in 1:max_iter
        J = ForwardDiff.jacobian(zzz, x)
        hlm_val = h(J, w, cost, x, l0)
        rho_val = rho_nonsquare(J, x, hlm_val, w,win, cost)
        println("iteracion = $i, ρ = $rho_val,  λ = $l0 ")
        if rho_val > 1e-7
            x += hlm_val
            l0 = max(l0 / 10, 1e-6)
        else
            l0 = min(l0 * 10, 1e7)
        end
        if norm(hlm_val) < eps()
            break
        end
    end
    return x
end

function rho_nonsquare_ten_years(J, x, h, w, cost)
    num = cost(x)' * w * cost(x) - cost(x + h)' * w * cost(x + h)
    den = abs(h' * (l0 * Diagonal(J'*w* J) * h + J' * w * cost(x)))
    return num / den
end
pro = [0,1,2,3]

function levenberg_marquardt_nonsquare_ten_years(x0, w, cost, zzz; l0=10.0, max_iter=100)
    x = copy(x0)
    for i in 1:max_iter
        J = ForwardDiff.jacobian(zzz, x)
        #J = J' * J + l0 * Diagonal(J' * J)
        hlm_val = h(J, w, cost, x, l0)
        rho_val = rho_nonsquare_ten_years(J, x, hlm_val, w, cost)
        println("iteracion = $i, ρ = $rho_val,  λ = $l0 , cost = ",sum(cost(x)))
        if rho_val > 0.001
            x += hlm_val
            l0 = max(l0 *0.5, 1e-7)
        else
            l0 = min(l0 * 1.5, 1e7)
        end
        if norm(hlm_val) < eps()
            break
        end
    end
    return x
end




# Initial guess
x0 = [1.0, 0.0001, 0.000001, 1.0]
J_system = ForwardDiff.jacobian(equation_system, x0)
# Run LM

@time begin
    result = levenberg_marquardt(x0, w, equation_system, zzz)
end

println("Levenberg-Marquardt result: ", result)
equation_system(result)


px = 15
py = 9
c = px*py - 4
ty = 11
tx = 15
Ly = 1.0
Lx = 10.0
epsx = Lx / (2*px)
epsy = Ly / (2py)

N = (px - 2)*(py - 2)
M_4x = px - 2
M_4y = py - 2
M = 2*M_4x + 2*M_4y
a = 2
b = 1
beta = Lx / Ly

weightss = zeros(2*(N+M)) ./ (2*(N+M))
weights3 = rand(3*(N+M),1) ./ (1*(N+M))

wx = weights3[1:(N+M)]
wy = weights3[(N+M)+1:2*(N+M)]
wp = weights3[2*(N+M)+1:end]

#rho = 1.0
mu = 1.0
vm = 1.0
Re = 1.0
pm = 10.0
Eu = 1.0

println("N+M = ", N+M)
println("Re = ", Re)
println("weights3 size = ", size(weights3))

function euclidian_distance(x, y, xyk, c)
    matrix = zeros(length(x)*length(y), c)
    number_matrix = zeros(length(x)*length(y), 2)
    for i in 1:length(x)
        for j in 1:length(y)
            m = j + (i-1)*length(y)
            matrix[m, :] .= sqrt.((x[i] .- xyk[:,1]).^2 .+ (y[j] .- xyk[:,2]).^2) .+ eps()
            number_matrix[m,1] = x[i]
            number_matrix[m,2] = y[j]
        end
    end
    return matrix, number_matrix
end

function points(x, y)
    matrix = zeros(length(x)*length(y), 2)
    boundary = zeros(2*length(y)-4 + 2*length(x), 2)
    interior = zeros((length(x)-2)*(length(y)-2), 2)
    h, u = 1, 1
    for i in 1:length(x)
        for j in 1:length(y)
            m = j + (i-1)*length(y)
            matrix[m,1] = x[i]
            matrix[m,2] = y[j]
            if i == 1 || i == length(x) || j == 1 || j == length(y)
                boundary[h, :] .= [x[i], y[j]]
                h += 1
            else
                interior[u, :] .= [x[i], y[j]]
                u += 1
            end
        end
    end
    return matrix, boundary, interior
end

# Grid and test points
x = range(0, Lx, length=px)
y = range(0, Ly, length=py)
X = repeat(x', py, 1)
Y = repeat(y, 1, px)

x_int = range(epsx, Lx - epsx, length=px-2)
y_int = range(epsy, Ly - epsy, length=py-2)
Xin = repeat(x_int', py-2, 1)
Yin = repeat(y_int, 1, px-2)

xf1, yf1 = x_int, [0.0]
xf2, yf2 = x_int, [Ly]
xf3, yf3 = [0.0], y_int
xf4, yf4 = [Lx], y_int

X1 = repeat(xf1', 1, 1)
Y1 = repeat(yf1, 1, length(xf1))

X2 = repeat(xf2', 1, 1)
Y2 = repeat(yf2, 1, length(xf2))

X3 = repeat(xf3', length(yf3), 1)
Y3 = repeat(yf3, 1, 1)

X4 = repeat(xf4', length(yf4), 1)
Y4 = repeat(yf4, 1, 1)

# Centers
xk = points(x_int, y_int)[1][:,1]
yk = points(x_int, y_int)[1][:,2]
xyk1 = hcat(xk, yk)

xykf1 = hcat(xf1, fill(0.0, length(xf1)))
xykf2 = hcat(xf2, fill(Ly, length(xf2)))
xykf3 = hcat(fill(0.0, length(yf3)), yf3)
xykf4 = hcat(fill(Lx, length(yf4)), yf4)

xyk1 = vcat(xyk1, xykf1, xykf2, xykf3, xykf4)

# Test points
x_test = range(0, Lx, length=tx)
y_test = range(0, Ly, length=ty)
Xt = repeat(x_test', ty, 1)
Yt = repeat(y_test, 1, tx)

x_vector = points(x_int, y_int)[1][:,1]
y_vector = points(x_int, y_int)[1][:,2]

println("x_vector size = ", size(x_vector))
#x_v2 = euclidian_distance(x, y, xyk1, c)

# Plot
scatter(Xin[:], Yin[:], marker=:square, color=:red, label="Interior")
scatter!(X1[:], Y1[:], marker=:square, color=:blue, label="Boundary Bottom")
scatter!(X2[:], Y2[:], marker=:square, color=:cyan, label="Boundary Top")
scatter!(X3[:], Y3[:], marker=:square, color=:green, label="Boundary Left")
scatter!(X4[:], Y4[:], marker=:square, color=:yellow, label="Boundary Right")

xlims!(-0.3 - Lx/px, Lx + Lx/px + 0.3)
ylims!(-0.2, Ly * 1.1)



# Thin Plate Spline (TPS) RBF and derivatives
function tps_rbf(d, a=a, b=b)
    tps = (d * b).^(2a) .* log.(b .* (d ))
    replace!(tps, NaN => 0.0)  # Handle NaN values
    return tps
end

function tpsdx(x_vector, d, a=a, b=b)
    diff = x_vector[:].*ones(1, size(d,2)) .- xyk1[:,1]'
    tps = (diff ./ d) .* (b)^(2a) .* d.^(2a - 1) .* (2a .* log.(b .* d ) .+ 1)
    replace!(tps, NaN => 0.0)  # Handle NaN values
    return tps
end


function tpsdy(y_vector, d, a=a, b=b)
    diff = y_vector[:].*ones(1, size(d,2)) .- xyk1[:,2]'
    tps = (b)^(2a) .* d.^(2a - 1) .* (2a .* log.(b .* d ) .+ 1) .* (diff ./ d)
    replace!(tps, NaN => 0.0)  # Handle NaN values
    return tps
end

function tpsdxx(x_vector, d, a=a, b=b)
    diff = x_vector[:].*ones(1, size(d,2)) .- xyk1[:,1]'
    term1 = ((d.^2 .- diff.^2) ./ d.^3) .* (b)^(2a) .* d.^(2a - 1) .* (2a .* log.(b .* (d )) .+ 1)
    term2 = (diff.^2 ./ d.^2) .* (b)^(2a) .* d.^(2a - 2) .* ((2a - 1) .* (2a .* log.(b .* (d )) .+ 1) .+ 2a)
    tps1 = replace!(term1 , NaN => 0.0)  # Handle NaN values
    tps2 = replace!(term2, NaN => 0.0)  # Handle NaN values
      # Handle NaN values
    return tps1 + tps2
end

function tpsdyy(y_vector, d, a=a, b=b)
    diff = y_vector[:].*ones(1, size(d,2)) .- xyk1[:,2]'
    term1 = ((d.^2 .- diff.^2) ./ d.^3) .* (b)^(2a) .* d.^(2a - 1) .* (2a .* log.(b .* (d )) .+ 1)
    term2 = (diff.^2 ./ d.^2) .* (b)^(2a) .* d.^(2a - 2) .* ((2a - 1) .* (2a .* log.(b .* (d )) .+ 1) .+ 2a)
    tps1 = replace!(term1 , NaN => 0.0)  # Handle NaN values
    tps2 = replace!(term2, NaN => 0.0)
    # Handle NaN values
    return tps1 + tps2
end


euclidian_dist_complete = euclidian_distance(x, y, xyk1, c)[1]
euclidian_dist_int = euclidian_distance(x_int, y_int, xyk1, c)[1]
euclidian_dist_centers = euclidian_distance(xyk1[:,1], xyk1[:,2], xyk1, c)[1]

euclidian_dist_border1 = euclidian_distance(xf1, yf1, xyk1, c)[1]
euclidian_dist_border2 = euclidian_distance(xf2, yf2, xyk1, c)[1]
euclidian_dist_border3 = euclidian_distance(xf3, yf3, xyk1, c)[1]
euclidian_dist_border4 = euclidian_distance(xf4, yf4, xyk1, c)[1]
a = 2
z = (euclidian_dist_int.^4 .*log.(euclidian_dist_int))
sum(z)
replace!(z, NaN => 0.0)
sum(z)

euclidian_dist_test = euclidian_distance(x_test, y_test, xyk1, c)[1]

x_vector_int = euclidian_distance(x_int, y_int, xyk1, c)[2][:,1]

y_vector_int = euclidian_distance(x_int, y_int, xyk1, c)[2][:,2]





# Matrix of RBFs and derivatives at interior points
g_int       = Array(tps_rbf(euclidian_dist_int))              # Φ
gx_int      = Array(tpsdx(x_vector_int, euclidian_dist_int))  # ∂Φ/∂x
gy_int      = Array(tpsdy(y_vector_int, euclidian_dist_int))  # ∂Φ/∂y
gxx_int     = Array(tpsdxx(x_vector_int, euclidian_dist_int)) # ∂²Φ/∂x²
gyy_int     = Array(tpsdyy(y_vector_int, euclidian_dist_int)) # ∂²Φ/∂y²

sum(gxx_int)
sum(gyy_int)
# Boundary conditions for b3 and b4
gb3         = Array(tps_rbf(euclidian_dist_border3))
gb4         = Array(tps_rbf(euclidian_dist_border4))
gx_b3       = Array(tpsdx(zeros(length(yf3)), euclidian_dist_border3))
gx_b4       = Array(tpsdx(fill(Lx, length(yf4)), euclidian_dist_border4))
gy_b3       = Array(tpsdy(yf3, euclidian_dist_border3))
gy_b4       = Array(tpsdy(yf4, euclidian_dist_border4))
gxx_b3      = Array(tpsdxx(zeros(length(yf3)), euclidian_dist_border3))
gxx_b4      = Array(tpsdxx(fill(Lx, length(yf4)), euclidian_dist_border4))
gyy_b3      = Array(tpsdyy(yf3, euclidian_dist_border3))
gyy_b4      = Array(tpsdyy(yf4, euclidian_dist_border4))

sum(gxx_b3)
sum(gyy_b3)

# Boundary values for top and bottom boundaries
gb1         = Array(tps_rbf(euclidian_dist_border1))
gb2         = Array(tps_rbf(euclidian_dist_border2))
gxx_b1      = Array(tpsdxx(xf1, euclidian_dist_border1))
gxx_b2      = Array(tpsdxx(xf2, euclidian_dist_border2))
gyy_b1      = Array(tpsdyy(zeros(length(xf1)), euclidian_dist_border1))
gyy_b2      = Array(tpsdyy(fill(Ly, length(xf2)), euclidian_dist_border2))
gy_b1       = Array(tpsdy(zeros(length(xf1)), euclidian_dist_border1))
gy_b2       = Array(tpsdy(fill(Ly, length(xf2)), euclidian_dist_border2))

sum(gb1)
sum(gb2)

function outer_dot_rows(A, B)
    C = zeros(size(A,1), size(A,2)^2)
    for i in 1:size(A, 1)
    C[i,:] = reshape(A[i,:] * transpose(B[i,:]),size(A,2)^2 )
    end
    return C
end

#for i in 1:size(g_int, 1)
#C[i,:] = reshape(g_int[i,:] * transpose(g_int[i,:]), size(g_int,2)^2)
#end

gb3
gx_int
phidphidx = (outer_dot_rows(g_int, gx_int))
phidphidy = outer_dot_rows(g_int, gy_int)
phidphidx_b3 = outer_dot_rows(gb3, gx_b3)
phidphidy_b3 = outer_dot_rows(gb3, gy_b3)
phidphidx_b4 = outer_dot_rows(gb4, gx_b4)
phidphidy_b4 = outer_dot_rows(gb4, gy_b4)

sum(gx_int)
sum(g_int)
sum(gb3)
sum(gb4)
sum(gy_b3)
sum(gy_b4)

sum(sum(phidphidx, dims=1))

function navier_stokes_residual!( weights_vec )
    F = similar(weights_vec, 3*N + 6*M_4y + 4*M_4x)
    wx = weights_vec[1:N+M]
    wy = weights_vec[N+M+1:2*(N+M)]
    wxiwxj = reshape((wx * transpose(wx)),:,1)
    wxiwyj = reshape((wx * transpose(wy)),:,1)
    wyiwxj = reshape((wy * transpose(wx)),:,1)
    wyiwyj = reshape((wy * transpose(wy)),:,1)
    F[1:N] = (1 / Re) * (-1 / beta^2 * (gxx_int*wx) - gyy_int * wx) + (phidphidx * wxiwxj) / beta + (phidphidy * wyiwxj) .+ (-pm) / beta
    F[N+1:2*N] = (1 / Re) * (-1 / beta^2 * (gxx_int * wy) - gyy_int * wy) + (phidphidx * wxiwyj) / beta + (phidphidy * wyiwyj)
    F[2*N+1:3*N] = (gx_int * wx) / beta + (gy_int * wy)
    #Condiciones de frontera entrada y salida
    F[3*N+1:3*N+M_4y] = (gx_b4 * wx) / beta + (gy_b4 * wy)
    F[3*N+M_4y+1:3*N+2*M_4y] = (gx_b3 * wx) / beta + (gy_b3 * wy)
    F[3*N+2*M_4y+1:3*N+3*M_4y] = (-1 / Re) * (1 / beta^2 * (gxx_b4 * wx) + gyy_b4 * wx) + (phidphidx_b4 * wxiwxj) / beta + (phidphidy_b4 * wyiwxj) .+ (-pm) / beta
    F[3*N+3*M_4y+1:3*N+4*M_4y] = (1 / Re) * (-1 / beta^2 * (gxx_b3 * wx) - gyy_b3 * wx) + (phidphidx_b3 * wxiwxj) / beta + (phidphidy_b3 * wyiwxj) .+ (-pm) / beta
    F[3*N+4*M_4y+1:3*N+5*M_4y] = (1 / Re) * (-1 / beta^2 * (gxx_b4 * wy) - gyy_b4 * wy) + (phidphidx_b4 * wxiwyj) / beta + (phidphidy_b4 * wyiwyj)
    F[3*N+5*M_4y+1:3*N+6*M_4y] = (1 / Re) * (-1 / beta^2 * (gxx_b3 * wy) - gyy_b3 * wy) + (phidphidx_b3 * wxiwyj) / beta + (phidphidy_b3 * wyiwyj)
    #Condiciones de frontera Dirichlet
    F[3*N+6*M_4y+1: 3*N+6*M_4y+ 1*M_4x] = gb2 * wx
    F[3*N+6*M_4y+M_4x+1:3*N+6*M_4y+2*M_4x] = gb1 * wx
    F[3*N+6*M_4y+2*M_4x+1:3*N+6*M_4y+3*M_4x] = gb2 * wy
    F[3*N+6*M_4y+3*M_4x+1:3*N+6*M_4y+4*M_4x] = gb1 * wy
    # Return the residual vector

    return F
end

function navier_stokes_residual_square!( weights_vec )
    F = similar(weights_vec, 2*N + 4*M_4y + 4*M_4x)
    wx = weights_vec[1:N+M]
    wy = weights_vec[N+M+1:2*(N+M)]
    wxiwxj = reshape((wx * transpose(wx)),:,1)
    wxiwyj = reshape((wx * transpose(wy)),:,1)
    wyiwxj = reshape((wy * transpose(wx)),:,1)
    wyiwyj = reshape((wy * transpose(wy)),:,1)
    F[1:N] = (1 / Re) * (-1 / beta^2 * (gxx_int*wx) - gyy_int * wx) - (phidphidy * wxiwyj) / beta + (phidphidy * wyiwxj) .+ (-pm) / beta
    F[N+1:2*N] = (1 / Re) * (-1 / beta^2 * (gxx_int * wy) - gyy_int * wy) + (phidphidx * wxiwyj) / beta + (phidphidy * wyiwyj)
    #F[2*N+1:3*N] = (gx_int * wx) / beta + (gy_int * wy)
    #Condiciones de frontera entrada y salida
    #F[2*N+1:3*N+M_4y] = (gx_b4 * wx) / beta + (gy_b4 * wy)
    #F[2*N+M_4y+1:3*N+2*M_4y] = (gx_b3 * wx) / beta + (gy_b3 * wy)
    F[2*N+1:2*N+1*M_4y] = (-1 / Re) * (1 / beta^2 * (gxx_b4 * wx) + gyy_b4 * wx) - (phidphidy_b4 * wxiwyj) / beta + (phidphidy_b4 * wyiwxj) .+ (-pm) / beta
    F[2*N+1*M_4y+1:2*N+2*M_4y] = (1 / Re) * (-1 / beta^2 * (gxx_b3 * wx) - gyy_b3 * wx) - (phidphidy_b3 * wxiwyj) / beta + (phidphidy_b3 * wyiwxj) .+ (-pm) / beta
    F[2*N+2*M_4y+1:2*N+3*M_4y] = (1 / Re) * (-1 / beta^2 * (gxx_b4 * wy) - gyy_b4 * wy) + (phidphidx_b4 * wxiwyj) / beta + (phidphidy_b4 * wyiwyj)
    F[2*N+3*M_4y+1:2*N+4*M_4y] = (1 / Re) * (-1 / beta^2 * (gxx_b3 * wy) - gyy_b3 * wy) + (phidphidx_b3 * wxiwyj) / beta + (phidphidy_b3 * wyiwyj)
    #Condiciones de frontera Dirichlet
    F[2*N+4*M_4y+1: 2*N+4*M_4y+ 1*M_4x] = gb2 * wx
    F[2*N+4*M_4y+M_4x+1:2*N+4*M_4y+2*M_4x] = gb1 * wx
    F[2*N+4*M_4y+2*M_4x+1:2*N+4*M_4y+3*M_4x] = gb2 * wy
    F[2*N+4*M_4y+3*M_4x+1:2*N+4*M_4y+4*M_4x] = gb1 * wy
    # Return the residual vector

    return F
end


navier_stokes_residual!(weightss)
eps()


function navier_stokess_system(weights_vec)
    
    F = similar(weights_vec, 3*N + 6*M_4y + 4*M_4x)
    wx = weights_vec[1:N+M]
    wy = weights_vec[N+M+1:2*(N+M)]
    wxiwxj = reshape((wx * transpose(wx)),:,1)
    wxiwyj = reshape((wx * transpose(wy)),:,1)
    wyiwxj = reshape((wy * transpose(wx)),:,1)
    wyiwyj = reshape((wy * transpose(wy)),:,1)
    F[1:N] = (1 / Re) * (-1 / beta^2 * (gxx_int * wx) - gyy_int * wx) + (phidphidx * wxiwxj) / beta + (phidphidy * wyiwxj)
    F[N+1:2*N] = (1 / Re) * (-1 / beta^2 * (gxx_int * wy) - gyy_int * wy) + (phidphidx * wxiwyj) / beta + (phidphidy * wyiwyj)
    F[2*N+1:3*N] = (gx_int * wx) / beta + (gy_int * wy)
    #Condiciones de frontera entrada y salida
    F[3*N+1:3*N+M_4y] = (gx_b4 * wx) / beta + (gy_b4 * wy)
    F[3*N+M_4y+1:3*N+2*M_4y] = (gx_b3 * wx) / beta + (gy_b3 * wy)
    F[3*N+2*M_4y+1:3*N+3*M_4y] = (-1 / Re) * (1 / beta^2 * (gxx_b4 * wx) + gyy_b4 * wx) + (phidphidx_b4 * wxiwxj) / beta + (phidphidy_b4 * wyiwxj)
    F[3*N+3*M_4y+1:3*N+4*M_4y] = (1 / Re) * (-1 / beta^2 * (gxx_b3 * wx) - gyy_b3 * wx) + (phidphidx_b3 * wxiwxj) / beta + (phidphidy_b3 * wyiwxj)
    F[3*N+4*M_4y+1:3*N+5*M_4y] = (1 / Re) * (-1 / beta^2 * (gxx_b4 * wy) - gyy_b4 * wy) + (phidphidx_b4 * wxiwyj) / beta + (phidphidy_b4 * wyiwyj)
    F[3*N+5*M_4y+1:3*N+6*M_4y] = (1 / Re) * (-1 / beta^2 * (gxx_b3 * wy) - gyy_b3 * wy) + (phidphidx_b3 * wxiwyj) / beta + (phidphidy_b3 * wyiwyj)
    #Condiciones de frontera Dirichlet
    F[3*N+6*M_4y+1: 3*N+6*M_4y+ 1*M_4x] = gb2 * wx
    F[3*N+6*M_4y+M_4x+1:3*N+6*M_4y+2*M_4x] = gb1 * wx
    F[3*N+6*M_4y+2*M_4x+1:3*N+6*M_4y+3*M_4x] = gb2 * wy
    F[3*N+6*M_4y+3*M_4x+1:3*N+6*M_4y+4*M_4x] = gb1 * wy
    # Return the residual vector

    return -F


end

function navier_stokes_system_square!( weights_vec )
    F = similar(weights_vec, 2*N + 4*M_4y + 4*M_4x)
    wx = weights_vec[1:N+M]
    wy = weights_vec[N+M+1:2*(N+M)]
    wxiwxj = reshape((wx * transpose(wx)),:,1)
    wxiwyj = reshape((wx * transpose(wy)),:,1)
    wyiwxj = reshape((wy * transpose(wx)),:,1)
    wyiwyj = reshape((wy * transpose(wy)),:,1)
    F[1:N] = (1 / Re) * (-1 / beta^2 * (gxx_int*wx) - gyy_int * wx) - (phidphidy * wxiwyj) / beta + (phidphidy * wyiwxj) #.+ (-pm) / beta
    F[N+1:2*N] = (1 / Re) * (-1 / beta^2 * (gxx_int * wy) - gyy_int * wy) + (phidphidx * wxiwyj) / beta + (phidphidy * wyiwyj)
    #F[2*N+1:3*N] = (gx_int * wx) / beta + (gy_int * wy)
    #Condiciones de frontera entrada y salida
    #F[2*N+1:3*N+M_4y] = (gx_b4 * wx) / beta + (gy_b4 * wy)
    #F[2*N+M_4y+1:3*N+2*M_4y] = (gx_b3 * wx) / beta + (gy_b3 * wy)
    F[2*N+1:2*N+1*M_4y] = (-1 / Re) * (1 / beta^2 * (gxx_b4 * wx) + gyy_b4 * wx) - (phidphidy_b4 * wxiwyj) / beta + (phidphidy_b4 * wyiwxj) #.+ (-pm) / beta
    F[2*N+1*M_4y+1:2*N+2*M_4y] = (1 / Re) * (-1 / beta^2 * (gxx_b3 * wx) - gyy_b3 * wx) - (phidphidy_b3 * wxiwyj) / beta + (phidphidy_b3 * wyiwxj) #.+ (-pm) / beta
    F[2*N+2*M_4y+1:2*N+3*M_4y] = (1 / Re) * (-1 / beta^2 * (gxx_b4 * wy) - gyy_b4 * wy) + (phidphidx_b4 * wxiwyj) / beta + (phidphidy_b4 * wyiwyj)
    F[2*N+3*M_4y+1:2*N+4*M_4y] = (1 / Re) * (-1 / beta^2 * (gxx_b3 * wy) - gyy_b3 * wy) + (phidphidx_b3 * wxiwyj) / beta + (phidphidy_b3 * wyiwyj)
    #Condiciones de frontera Dirichlet
    F[2*N+4*M_4y+1: 2*N+4*M_4y+ 1*M_4x] = gb2 * wx
    F[2*N+4*M_4y+M_4x+1:2*N+4*M_4y+2*M_4x] = gb1 * wx
    F[2*N+4*M_4y+2*M_4x+1:2*N+4*M_4y+3*M_4x] = gb2 * wy
    F[2*N+4*M_4y+3*M_4x+1:2*N+4*M_4y+4*M_4x] = gb1 * wy
    # Return the residual vector

    return -F
end


size(navier_stokes_system_square!(weightss))
size(navier_stokes_residual!(weightss),1)

size(navier_stokes_residual_square!(weightss))

navier_stokes_residual!(weightss)
#sum(navier_stokes_residual!(weightss))

wns = Diagonal(ones(size(navier_stokes_residual!(weightss),1)) )
wns_square = Diagonal(ones(size(navier_stokes_residual_square!(weightss),1)) )
#win = Diagonal(ones(size(weightss,1)) )
J_NS =  ForwardDiff.jacobian(navier_stokess_system, weightss)



sJNS = J_NS'*wns* J_NS

size(J_NS)

J = J_NS' * J_NS + l0 * Diagonal(J_NS' * J_NS)

size(J)


minimum(sJNS), maximum(sJNS)
maximum(J_NS), minimum(J_NS)

Diagonal(J_NS'*wns* J_NS)

weightss
hns = h(J_NS, wns, navier_stokes_residual!, weightss, l0)
rho_nonsquare_ten_years(J_NS, weightss,hns , wns, navier_stokes_residual!)

J_square = ForwardDiff.jacobian(navier_stokes_residual_square!, weightss)
J_s_square = ForwardDiff.jacobian(navier_stokes_system_square!, weightss)


@time begin
    result = levenberg_marquardt_nonsquare_ten_years(weightss, wns_square, navier_stokes_residual_square!, navier_stokes_system_square!)
end


sum(navier_stokes_residual_square!(result))

Re = 100

@time begin
    result = levenberg_marquardt_nonsquare_ten_years(result, wns, navier_stokes_residual!, navier_stokess_system)
end


vx =beta* tps_rbf(euclidian_dist_int)* result[1:N+M]
vy = tps_rbf(euclidian_dist_int)*result[N+M+1:2*(N+M)]
maximum(vx), minimum(vx)
maximum(vy), minimum(vy)


# Assume x_int and y_int are 1D arrays
X = repeat(x_int, inner=length(y_int))
Y = repeat(y_int, outer=length(x_int))

# Compute velocity fields (already done)
# vx = beta * tps_rbf(euclidian_dist_int) * result[1:N+M]
# vy = tps_rbf(euclidian_dist_int) * result[N+M+1:2*(N+M)]

# Plot quiver

realvx = y-> -pm/2* Re*(y^2 - Ly*y)
realvy = y-> -pm/2* Re*(y^2 - Ly*y)*0
realvxi = realvx.(Y)
realvyi = realvy.(Y)
quiver(X, Y, quiver=(vx, vy), aspect_ratio=1, color=:blue, title="Velocity Field", xlabel="x", ylabel="y")
quiver!(X, Y, quiver=(realvxi, realvyi), color=:red, label="Real Velocity", alpha=0.5)



100*(sum(abs.((vx-realvxi)/realvxi))/size(vx,1))
100*(sum((((vx-realvxi))/size(vx,1)).^2 ) ^0.5)

