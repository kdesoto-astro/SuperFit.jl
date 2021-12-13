using Test
using SuperFit
include("integration_test_functions.jl")

@testset "Testing SuperFit.jl" begin
    
    @testset "Utility function unit tests" begin
        @test median_absolute_deviation([1.0, 3.0, 5.0, 7.0, 9.0]) == 2.0
        @test median_absolute_deviation([1.0, 1.0, 1.0, 1.0, 1.0]) == 0.
        
        ZTF_MJD = [1., 2., 3.]
        ZTF_PSF = [19., 20., 21.]
        ZTF_PSFerr = [1., 1., 1.]
        obs = (;ZTF_MJD, ZTF_PSF, ZTF_PSFerr)
        flux_obs = convert_mags_to_flux(obs, 22.)
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
        t = [0.5, 5.0]
        @test flux_map(t, params) ≈ [0.488721, 0.578030]
        
    end;
    
    @testset "Integration tests for low-noise simulated data" begin 
        @test test_simulated_data(100, 100., 0.002, 6., 70., 0.1, 58840., 15., 50., 0.5)
        @test test_simulated_data(100, 100., 0.007, 6., 70., 0.9, 59100., 5., 100., 0.5)
        @test test_simulated_data(100, 100., 0.005, 5., 60., 0.3, 59100., 25., 150., 0.5)
        @test test_simulated_data(100, 100., 0.009, 15., 80., 0.1, 58500., 1., 290., 0.5) #tricky
    end;

end;
