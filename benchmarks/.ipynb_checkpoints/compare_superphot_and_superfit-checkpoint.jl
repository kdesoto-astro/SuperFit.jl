import Pkg
Pkg.activate("..")
Pkg.instantiate()
Pkg.precompile()

using Distributed

machinefilename = ENV["PBS_NODEFILE"]
println(ENV["PBS_NUM_NODES"])
println(ENV["PBS_NUM_PPN"])
machinespecs = readlines( machinefilename )
threads_per_process = parse(Int64, ENV["PBS_NUM_PPN"])
addprocs(machinespecs[1:threads_per_process:end], exeflags="-t " * string(threads_per_process))


@everywhere import Pkg
#@everywhere Pkg.offline(true)
@everywhere Pkg.UPDATED_REGISTRY_THIS_SESSION[] = true 
@everywhere Pkg.activate("..")

@everywhere using Dates

println(string("starting imports", now()))

@everywhere using SuperFit
@everywhere using Turing
@everywhere using PyCall
@everywhere using Conda
@everywhere using Formatting

#ENV["PYTHON"]=""
#Pkg.build("PyCall")
#Conda.add("pymc3")
#Conda.add("astropy")
@everywhere pushfirst!(PyVector(pyimport("sys")."path"), "")
@everywhere superphot = pyimport("superphot_fit")

println(string("finished all imports", now()))

@everywhere function run_both_samplers(filename, timing_file)
    println(string("entered run both samplers function", now()))
    outdir = "../../compared_models"
    sn_name = split(Base.Filesystem.basename(filename), ".")[1]
    outfile1 = joinpath(outdir, string(sn_name, "{}", "_phot"))
    outfile2 = joinpath(outdir, string(sn_name, "{}", "_fit", ".jls"))
    light_curve = SuperFit.read_light_curve(filename)
    println(string("read lightcurve", now()))
    #if args.zmin is not None and light_curve.meta['REDSHIFT'] <= args.zmin:
    #    raise ValueError(f'Skipping file with redshift {light_curve.meta["REDSHIFT"]}: {filename}')
    t = SuperFit.select_event_data(light_curve)
    filters = SuperFit.FILTERS
    algorithm = NUTS()
    iterations = SuperFit.ITERATIONS
    walkers = 25
    #TODO: have better filter list checl
    println(string("about to start loop", now()))
    for fltr in filters
        println(format("STARTING FILTER {}", fltr))
        obs_mags = filter(row -> row.ZTF_filter == fltr, t)
        obs_time, obs_flux, obs_unc = SuperFit.convert_mags_to_flux(obs_mags, SuperFit.ZEROPOINT_MAG)
        println(string("convert mag to flux", now()))
        outfile_phot = format(outfile1, string("_",  fltr))
        model_phot, params = superphot.setup_model1(obs_time, obs_flux, obs_unc)
        println(string("setup phot model", now()))
        runtime_phot = @elapsed superphot.sample_or_load_trace(model_phot, outfile_phot)
        println(string("sampled superphot", now()))
        model = SuperFit.setup_model(obs_time, obs_flux, obs_unc)
        println(string("setup fit model", now()))
        outfile_fit = format(outfile2, string("_", fltr))
        runtime_fit = @elapsed SuperFit.sample_or_load_trace(
            model,
            outfile_fit,
            force=true,
            algorithm=algorithm,
            iterations=iterations,
            walkers=walkers
        )
        println(string("sampled superfit", now()))
        open(timing_file, "a+") do tf
            write(tf, string(sn_name, ",", runtime_phot, ",", runtime_fit))
            write(tf, "\n")
        end
        println(string("wrote to file", now()))
    end

end
    
@everywhere function main()
    println(string("entered main loop", now()))
    println(Threads.nthreads())
    ztf_dir = "../../project-kdesoto-psu/ztf_data/filtered_data_20"
    timing_file = "superphot_superfit_comparison.txt"
    all_ztf_files = readdir(ztf_dir; join=true)
    println(string("found files", now()))
    valid_filenames = Vector{String}()
    for filename in all_ztf_files
        if occursin(".txt", filename)
            push!(valid_filenames, filename)
            #run_both_samplers(filename, timing_file)
        end
    end
    println(string("beginning pmap", now()))
    pmap(x -> run_both_samplers(x, timing_file), valid_filenames)
end
    
        
main()

