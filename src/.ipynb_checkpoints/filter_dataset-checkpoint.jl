
using CSV, DataFrames, Plots

function merge_series!(sp1::Plots.Subplot, sp2::Plots.Subplot)
    append!(sp1.series_list, sp2.series_list)
    Plots.expand_extrema!(sp1[:xaxis], xlims(sp2))
    Plots.expand_extrema!(sp1[:yaxis], ylims(sp2))
    Plots.expand_extrema!(sp1[:zaxis], zlims(sp2))
    return sp1
end

function merge_series!(plt, plts...)
    for (i, sp) in enumerate(plt.subplots)
        for other_plt in plts
            if i in eachindex(other_plt.subplots)
                merge_series!(sp, other_plt[i])
            end
        end
    end
    return plt
end

ENV["GKSwstype"] = "100"
MIN_DATAPOINTS_PER_BAND = 20
SUPERNOVA_TYPES = ["SN Ia", "SN II", "SN 1b", "SN 1c", "SN 1b/c", "SN IIn", "SLSN-I", "SLSN-II"]

ztf_transient_summary = "../ztf_data/all_ztf_transients.csv"
all_lightcurve_data = "../ztf_data/lightcurves/ztf/"
output_dir = string("../ztf_data/filtered_data_",MIN_DATAPOINTS_PER_BAND)

mkpath(output_dir)

summary_df = CSV.read(ztf_transient_summary, DataFrame)
filtered_df = filter(row -> !ismissing(row.type) && row.type in SUPERNOVA_TYPES, summary_df)
ztf_names = filtered_df[:, 24]

for ztf_name = ztf_names
    ztf_filepath = joinpath(all_lightcurve_data, string(ztf_name,"_ztf.txt"))
    ztf_photometry = CSV.read(ztf_filepath, DataFrame, delim=' ')
    r_entries = filter(row -> row.ZTF_filter == "r", ztf_photometry)
    g_entries = filter(row -> row.ZTF_filter == "g", ztf_photometry)
    if nrow(r_entries) >= MIN_DATAPOINTS_PER_BAND && nrow(g_entries) >= MIN_DATAPOINTS_PER_BAND
        ENV["GKSwstype"] = "100"
        println(g_entries.ZTF_MJD)
        println(g_entries.ZTF_PSF)
        p1 = plot(r_entries.ZTF_MJD, r_entries.ZTF_PSF, seriestype = :scatter, yflip=true, color=1)
        p2 = plot(g_entries.ZTF_MJD, g_entries.ZTF_PSF, seriestype = :scatter, yflip=true, color=2)
        plt = merge_series!(p1, p2)
        savefig(plt, string("lc/sn_lc_",ztf_name,".png"))
        cp(ztf_filepath, joinpath(output_dir, string(ztf_name,".txt")), force=true)
        println(string(ztf_name," added"))
    end
end


