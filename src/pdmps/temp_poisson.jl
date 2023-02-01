using Distributions: MvNormal, logpdf, rand, Exponential
using LinearAlgebra: norm
using Random: randn, rand, Random

# Tempory Files that will be replace by methods in generic_couplings and poisson

function lin_bound(a, b)
    u = rand()
    if (b > 0)
        if (a < 0)
          return -a/b + lin_bound(0.0, b);
        else # a >= 0
          return -a/b + sqrt(a^2/b^2 - 2 * log(u)/b);
        end
      elseif (b == 0) # degenerate case
        if (a < 0)
          return Inf;
        else # a >= 0
          return -log(u)/a;
        end
      else # b <= 0
        if (a <= 0)
          return Inf;
        else # a > 0
          y = -log(u); t1=-a/b;
          if (y >= a * t1 + b *t1^2/2)
            return Inf;
          else
            return -a/b - sqrt(a^2/b^2 + 2 * y /b);
          end
        end
      end
end

function coupling_exp(λ, Δt)
  rp1() = rand(Exponential()) / λ
  rp2() = rand(Exponential()) / λ + Δt

  logdp1(x) = logpdf(Exponential(λ), x)
  logdp2(x) = logpdf(Exponential(λ), x - Δt)

  X = rp1()
  Y = copy(X)
  acc = (rand() < exp(logdp2(X) - logdp1(X)))
  coupled = true

  while !acc
      coupled = false
      Y = rp2()
      V = rand()
      acc = (exp(logdp1(Y) - logdp2(Y)) < V)
  end
  Y = Y - Δt
  return coupled, (X, Y)
end

