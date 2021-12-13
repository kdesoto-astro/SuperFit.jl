using Test
using SuperFit
include("integration_test_functions.jl")

@testset "Testing SuperFit.jl" begin
    
    @testset "Utility function unit tests" begin
        @test SuperFit.median_absolute_deviation([1.0, 3.0, 5.0, 7.0, 9.0]) == 2.0
        @test SuperFit.median_absolute_deviation([1.0, 1.0, 1.0, 1.0, 1.0]) == 0.
        
        ZTF_MJD = [1., 2., 3.]
        ZTF_PSF = [19., 20., 21.]
        ZTF_PSFerr = [1., 1., 1.]
        obs = (;ZTF_MJD, ZTF_PSF, ZTF_PSFerr)
        flux_obs = SuperFit.convert_mags_to_flux(obs, 22.)
        @test flux_obs[2] ≈ [10^(3. / 2.5), 10^(2. / 2.5), 10^(1. / 2.5)]
        @test flux_obs[3] ≈ [log(10.)*10^(3. / 2.5) / 2.5, 
            log(10.)*10^(2. / 2.5) / 2.5, 
            log(10.)*10^(1. / 2.5) / 2.5]
        
        params = (A=1.0,
            beta=0.005,
            gamma_1=3.0,
            gamma_2=60.0,
            gamma_switch=0.1,
            t_0=1.0,
            tau_rise=10.0,
            tau_fall=50.0
        )
        @test isapprox(SuperFit.flux_map(0.5, params), 0.488721, atol=.000001)
        @test isapprox(SuperFit.flux_map(5.0, params), 0.578030, atol=.000001)
        
    end;
    
    @testset "Integration tests for low-noise simulated data" begin 
        @test test_simulated_data(100, (A=100., beta=0.002,
                gamma_1=6., gamma_2=70., gamma_switch=0.1,
                t_0=58840., tau_rise=15., tau_fall=50.), 0.01)
        @test test_simulated_data(100, (A=100., beta=0.007,
                gamma_1=6., gamma_2=70., gamma_switch=0.9,
                t_0=59100., tau_rise=5., tau_fall=100.), 0.01)
        @test test_simulated_data(100, (A=100., beta=0.005, 
                gamma_1=5., gamma_2=60., gamma_switch=0.3,
                t_0=59100., tau_rise=25., tau_fall=150.), 0.01)
        @test test_simulated_data(100, (A=100., beta=0.009,
                gamma_1=15., gamma_2=80., gamma_switch=0.1,
                t_0=58500., tau_rise=1., tau_fall=290.), 0.01) #tricky
    end;

end;
