using Test
include("../src/generic_couplings/exponentials.jl")


function test_coupling()
    # test that the probability of t_1 = t_2 + shift is the same for different shifts
    rate = 1.0
    mode = "independent"
    mixture_weight = exp(-rate)
    shift_values = [0.1, 0.5, 1.0, 2.0]
    tol = 1e-2
    for shift in shift_values
        coupled_count = 0
        uncoupled_count = 0
        N = 100000
        for i in 1:N
            t1, t2, coupled = coupling(rate, shift, mode)
            if coupled
                coupled_count += 1
                @test abs(t1 - t2 - shift) < tol
            else
                uncoupled_count += 1
            end
        end
        coupled_prob = coupled_count / N
        uncoupled_prob = uncoupled_count / N
        mixture_weight = exp(-shift * rate)

        expected_coupled_prob = mixture_weight
        expected_uncoupled_prob = 1.0 - mixture_weight
        @test abs(coupled_prob - expected_coupled_prob) < tol
        @test abs(uncoupled_prob - expected_uncoupled_prob) < tol
    end
end


function marginal_tests()
    rate = 1.0
    shift = 1.0
    n_samples = 10000
    t_1 = zeros(n_samples)
    t_2 = zeros(n_samples)

    @testset "Coupling of exponential distributions" begin
        @testset "Independent mode" begin
            for i in 1:n_samples
                a, b, _ = coupling(rate, shift, "independent")
                t_1[i] = a
                t_2[i] = b
            end
            @test isapprox(mean(t_1), shift; atol=0.1)
            @test isapprox(mean(t_2), 0.0; atol=0.1)
            @test isapprox(var(t_1), 1.0/rate^2; atol=0.1)
            @test isapprox(var(t_2), 1.0/rate^2; atol=0.1)
            @test ks_2samp(t_1, t_2 + shift).statistic >= 0.0
            @test ks_2samp(t_1, t_2 + shift).pvalue <= 0.05
        end

        @testset "CRN mode" begin
            for i in 1:n_samples
                a, b, _ = coupling(rate, shift, "crn")
                t_1[i] = a
                t_2[i] = b
            end
            @test isapprox(mean(t_1), shift; atol=0.1)
            @test isapprox(mean(t_2), 0.0; atol=0.1)
            @test isapprox(var(t_1), 1.0/rate^2; atol=0.1)
            @test isapprox(var(t_2), 1.0/rate^2; atol=0.1)
            @test ks_2samp(t_1, t_2 + shift).statistic >= 0.0
            @test ks_2samp(t_1, t_2 + shift).pvalue <= 0.05
        end

        @testset "Antithetic mode" begin
            for i in 1:n_samples
                a, b, _ = coupling(rate, shift, "antithetic")
                t_1[i] = a
                t_2[i] = b
            end
            @test isapprox(mean(t_1), shift; atol=0.1)
            @test isapprox(mean(t_2), 0.0; atol=0.1)
            @test isapprox(var(t_1), 1.0/rate^2; atol=0.1)
            @test isapprox(var(t_2), 1.0/rate^2; atol=0.1)
            @test ks_2samp(t_1, t_2 + shift).statistic >= 0.0
            @test ks_2samp(t_1, t_2 + shift).pvalue <= 0.05
        end
    end
end


@testset "Tests" begin
    @testset "Checking coupling proba is the same for all methods" begin
        test_coupling()
    end
    @testset "Checking marginals are correct" begin
        test_coupling()
    end
end