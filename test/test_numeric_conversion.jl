using QuantumCumulants
using QuantumOpticsBase
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

# test LazyKet for initial product state
h1 = FockSpace(:fock1)
h2 = FockSpace(:fock2)
h3 = NLevelSpace(:atom,3)
h = h1 ⊗ h2 ⊗ h3

@qnumbers a1::Destroy(h,1) a2::Destroy(h,2)
σ(i,j) = Transition(h, :σ, i, j)

b1 = FockBasis(200)
b2 = FockBasis(250)
b3 = NLevelBasis(3)
b = b1 ⊗ b2 ⊗ b3

a1a2n = to_numeric_lazy(a1*a2,b)
a1n = to_numeric_lazy(a1,b)
a2n = to_numeric_lazy(a2,b)

ψ1 = coherentstate(b1, 2.0)
ψ2 = coherentstate(b2, 3.0)
# ψ1_ = coherentstate(b1, 5)
# ψ2_ = coherentstate(b2, 6)
ψ3 = nlevelstate(b3,2)
ψ = LazyKet(b, (ψ1, ψ2, ψ3))

# ψ12 = ψ1⊗ψ2 + ψ1_⊗ψ2_
# ψ_ = LazyKet(b, (ψ12, ψ3))

a1_ = numeric_average(a1,ψ)
a2_ = numeric_average(a2,ψ)
s22_ = numeric_average(σ(2,2),ψ)
a1_a2_ = numeric_average(a1*a2,ψ)

@test a1_a2_ == a1_*a2_
@test s22_ == 1.0

### test initial condition ###


end # testset
