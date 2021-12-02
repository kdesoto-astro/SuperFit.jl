using Test
include("integration_test_functions.jl")

@testset "Testing Julia Supernova-fit" begin
    
    @testset "Checking fit for low-noise simulated data" begin 
        #using .FitSNTestFunctions
        @test test_simulated_data(100, 100., 0.002, 6., 70., 0.1, 58840., 15., 50., 0.5)
        @test test_simulated_data(100, 100., 0.007, 6., 70., 0.9, 59100., 5., 100., 0.5)
        @test test_simulated_data(100, 100., 0.005, 5., 60., 0.3, 59100., 25., 150., 0.5)
        @test test_simulated_data(100, 100., 0.009, 15., 80., 0.1, 58500., 1., 290., 0.5) #tricky
    end;

end;
