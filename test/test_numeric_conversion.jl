using QuantumCumulants
using QuantumOpticsBase
using ModelingToolkit
using OrdinaryDiffEq
using Test
using Random; Random.seed!(0)

@testset "numeric-conversion" begin

# Test fock basis conversion
hfock = FockSpace(:fock)
@qnumbers a::Destroy(hfock)
bfock = FockBasis(7)
@test to_numeric(a, bfock) == destroy(bfock)
@test to_numeric(a', bfock) == create(bfock)

# NLevelSpace conversion
hnlevel = NLevelSpace(:nlevel, 3)
σ(i,j) = Transition(hnlevel, :σ, i, j)
bnlevel = NLevelBasis(3)
for i=1:3, j=1:3
    op = σ(i,j)
    @test to_numeric(op, bnlevel) == transition(bnlevel, i, j)
end

# with symbolic levels
levels = (:g, :e, :a)
hnlevel_sym = NLevelSpace(:nlevel_sym, levels)
σ_sym(i,j) = Transition(hnlevel_sym, :σ, i, j)
@test_throws ArgumentError to_numeric(σ_sym(:e,:g), bnlevel)
level_map = Dict((levels .=> (1,2,3))...)
for i=1:3, j=1:3
    lvl1 = levels[i]
    lvl2 = levels[j]
    op = σ_sym(lvl1, lvl2)
    @test to_numeric(op, bnlevel; level_map=level_map) == transition(bnlevel, i, j)
end

# On composite bases
hprod = hfock ⊗ hnlevel
a = Destroy(hprod, :a)
σprod(i,j) = Transition(hprod, :σ, i, j)
bprod = bfock ⊗ bnlevel
for i=1:3, j=1:3
    op1 = a*σprod(i,j)
    op2 = a'*σprod(i,j)
    @test to_numeric(op1, bprod) == destroy(bfock) ⊗ transition(bnlevel, i, j)
    @test to_numeric(op2, bprod) == create(bfock) ⊗ transition(bnlevel, i, j)
end

@test to_numeric(a'*a, bprod) ≈ number(bfock) ⊗ one(bnlevel)

# Composite basis with symbolic levels
σsym_prod(i,j) = Transition(hfock ⊗ hnlevel_sym, :σ, i, j)
a = Destroy(hfock ⊗ hnlevel_sym, :a)
@test_throws ArgumentError to_numeric(a*σsym_prod(:e,:g), bprod)
for i=1:3, j=1:3
    op1 = a*σsym_prod(levels[i],levels[j])
    op2 = a'*σsym_prod(levels[i],levels[j])
    @test to_numeric(op1, bprod; level_map=level_map) == destroy(bfock) ⊗ transition(bnlevel, i, j)
    @test to_numeric(op2, bprod; level_map=level_map) == create(bfock) ⊗ transition(bnlevel, i, j)
end

# Numeric average values
α = 0.1 + 0.2im
ψ = coherentstate(bfock, α)
a = Destroy(hfock, :a)
@test numeric_average(a, ψ) ≈ numeric_average(average(a), ψ) ≈ α
@test numeric_average(a'*a, ψ) ≈ numeric_average(average(a'*a), ψ) ≈ abs2(α)

ψprod = ψ ⊗ nlevelstate(bnlevel, 1)
@test_throws ArgumentError numeric_average(σsym_prod(:e,:g), ψprod)
idfock = one(bfock)
for i=1:3, j=1:3
    op = σprod(i, j)
    op_sym = σsym_prod(levels[i],levels[j])
    op_num = idfock ⊗ transition(bnlevel, i, j)
    @test numeric_average(op, ψprod) ≈ expect(op_num, ψprod)
    @test numeric_average(op_sym, ψprod; level_map=level_map) ≈ expect(op_num, ψprod)
end


# Initial values in actual equations
levels = (:g,:e)
h = FockSpace(:cavity) ⊗ NLevelSpace(:atom, levels)
a = Destroy(h,:a)
s(i,j) = Transition(h, :σ, i, j)

@cnumbers Δ g κ γ η

H = Δ*a'*a + g*(a'*s(:g,:e) + a*s(:e,:g)) + η*(a + a')
ops = [a,s(:g,:e),a'*a,s(:e,:e),a'*s(:g,:e)]
eqs = meanfield(ops,H,[a];rates=[κ],order=2)

bcav = FockBasis(10)
batom = NLevelBasis(2)
b = bcav ⊗ batom
ψ0 = randstate(b)

level_map = Dict((levels .=> [1,2])...)
u0 = initial_values(eqs, ψ0; level_map=level_map)

@test u0[1] ≈ expect(destroy(bcav) ⊗ one(batom), ψ0)
@test u0[2] ≈ expect(one(bcav) ⊗ transition(batom, 1, 2), ψ0)
@test u0[3] ≈ expect(number(bcav) ⊗ one(batom), ψ0)
@test u0[4] ≈ expect(one(bcav) ⊗ transition(batom, 2, 2), ψ0)
@test u0[5] ≈ expect(create(bcav) ⊗ transition(batom, 1, 2), ψ0)

# Test sufficiently large hilbert space; from issue #109
hfock = FockSpace(:fock)
@qnumbers a::Destroy(hfock)
bfock = FockBasis(100)

diff = (2*create(bfock)+2*destroy(bfock)) - to_numeric((2*(a)+2*(a')), bfock)
@test isequal(2*create(bfock)+2*destroy(bfock),to_numeric((2*(a)+2*(a')), bfock))
@test iszero(diff)

@test isequal(to_numeric(2*a, bfock),2*to_numeric(a, bfock))
@test iszero(to_numeric(2*a, bfock) - 2*to_numeric(a, bfock))

# Test indexed initial state (superradiant pulse)
order = 2 #order of the cumulant expansion
@cnumbers κ g Γ Δ N
hc = FockSpace(:cavity)
ha = NLevelSpace(:atom, 2)
h = hc ⊗ ha
i = Index(h,:i,N,ha)
j = Index(h,:j,N,ha)
k = Index(h,:k,N,ha)
@qnumbers a::Destroy(h,1)
σ(α,β,k) = IndexedOperator(Transition(h,:σ,α,β,2), k)
bc = FockBasis(3)
ba = NLevelBasis(2)
b = tensor(bc, [ba for i=1:order]...)
ψc = fockstate(bc, 0)
ψa = normalize(nlevelstate(ba,1) + nlevelstate(ba,2))
ψ = tensor(ψc, [ψa for i=1:order]...)
a_ = embed(b,1,destroy(bc))
σ_(i,j,k) = embed(b,1+k,transition(ba,i,j))
ranges=[1,2]
@test to_numeric(σ(1,2,1),b; ranges=ranges) == σ_(1,2,1)
@test to_numeric(σ(2,2,2),b; ranges=ranges) == σ_(2,2,2)
@test to_numeric(a,b; ranges=ranges) == a_
@test to_numeric(a*σ(2,2,2),b; ranges=ranges) == σ_(2,2,2)*a_
@test numeric_average(σ(2,2,2), ψ; ranges=ranges) ≈ 0.5
@test numeric_average(average(σ(2,2,1)), ψ; ranges=ranges) ≈ 0.5
@test numeric_average(average(a'a), ψ; ranges=ranges) ≈ 0.0
@test numeric_average(average(a*σ(2,2,1)), ψ; ranges=ranges) ≈ 0.0
@test_throws ArgumentError numeric_average(average(a'a), ψ)
# Hamiltonian
H = -Δ*a'a + g*(Σ(a'*σ(1,2,i),i) + Σ(a*σ(2,1,i),i))
J = [a,σ(1,2,i)]
rates = [κ, Γ]
ops = [a,σ(2,2,j)]
eqs = meanfield(ops,H,J;rates=rates,order=order)
eqs_c = complete(eqs)
eqs_sc = scale(eqs_c)
@named sys = ODESystem(eqs_sc)
@test_throws ArgumentError initial_values(eqs_sc, ψ)
u0 = initial_values(eqs_sc, ψ; ranges=ranges)
N_ = 2e5
Γ_ = 1.0 #Γ=1mHz
Δ_ = 2500Γ_ #Δ=2.5Hz
g_ = 1000Γ_ #g=1Hz
κ_ = 5e6*Γ_ #κ=5kHz
ps = [N, Δ, g, κ, Γ]
p0 = [N_, Δ_, g_, κ_, Γ_]

prob = ODEProblem(sys,u0,(0.0, 1e-4/Γ_), ps.=>p0)
@test solve(prob, Tsit5()) isa ODESolution

end # testset
