#!/usr/bin/env julia
"""
    generate_performance_profiles.jl
    
Generate performance profiles from CSV results for Examples 5.1, 5.2, 5.3.

Features:
- Scenario separation (A vs B+C)
- Method filtering (all vs approximate)
- Automatic PDF saving to plots directory
- Proper failure handling (Inf for non-converged)
- Batch processing for all examples

Usage:
    include("test/generate_performance_profiles.jl")
    process_example("5.1")
    process_all_examples()
"""

using DrWatson
@quickactivate :CRM_VIP

using CSV
using DataFrames
using Statistics: median
using PGFPlotsX, Plots
using BenchmarkProfiles

#=============================================================================
    METHOD DEFINITIONS
=============================================================================#

# ALL methods (Scenario A)
const METHODS_ALL = [
    "Extragradient",
    "Malitsky (Adaptive)",
    "Bello-Cruz Iusem 2010 (BI1)",
    "CRM-VIP Alg1",
    "Bello-Cruz Iusem 2012 (BI2)",
    "CRM-VIP Alg2"
]

# APPROXIMATE methods only (Scenarios B+C)
const METHODS_APPROX_ONLY = [
    "Bello-Cruz Iusem 2010 (BI1)",
    "CRM-VIP Alg1",
    "Bello-Cruz Iusem 2012 (BI2)",
    "CRM-VIP Alg2"
]

#=============================================================================
    PERFORMANCE PROFILE UTILITIES
=============================================================================#

"""
    build_performance_matrices(df, metric::Symbol)

Build matrix for performance profile from DataFrame.
Returns (matrix, method_names).

Failures (non-convergence) are marked with Inf as per BenchmarkProfiles.jl:
"Failures on a given problem are represented by a negative value, 
an infinite value, or NaN."
"""
function build_performance_matrices(df::DataFrame, metric::Symbol)
    df_work = copy(df)

    # 1. GERAR ID ÚNICO PARA CADA EXECUÇÃO (CRUCIAL)
    # O PerformanceProfile precisa comparar a execução X do Método A com a execução X do Método B.
    # Como seus dados podem ter várias seeds para o mesmo tamanho (ex: 100_10), 
    # criamos um índice (RunID) para diferenciar: 100_10_1, 100_10_2, etc.

    # Agrupa para numerar as ocorrências de cada problema por método
    transform!(groupby(df_work, [:Method, :Dimension, :Num_Ellipsoids]),
        eachindex => :RunID)

    # Cria o ID único que combina o problema e o número da execução
    df_work.UniqueProblemID = string.(df_work.Dimension, "_",
        df_work.Num_Ellipsoids, "_",
        df_work.RunID)

    # Listas ordenadas para garantir o alinhamento da matriz
    methods = sort(unique(df_work.Method))
    problems = sort(unique(df_work.UniqueProblemID))

    n_problems = length(problems)
    n_methods = length(methods)

    if n_problems == 0 || n_methods == 0
        return nothing, nothing
    end

    # Inicializa tudo com Inf (Falha por padrão)
    perf_matrix = fill(Inf, n_problems, n_methods)

    # 2. PREENCHIMENTO DIRETO (SEM FILTROS QUE ESCONDEM DADOS)
    for (i, prob) in enumerate(problems)
        for (j, meth) in enumerate(methods)
            # Busca a linha exata dessa instância (Prob + Seed) para esse método
            row_idx = findfirst(r -> r.UniqueProblemID == prob && r.Method == meth, eachrow(df_work))

            if !isnothing(row_idx)
                r = df_work[row_idx, :]

                # LÓGICA DIRETA:
                # É Solved? -> Pega o valor.
                # É Tired/Qualquer outra coisa? -> Vira Inf.
                if string(r.Status) in ["Solved", "Solved_xₖ", "Solved_yₖ"]
                    perf_matrix[i, j] = r[metric]
                else
                    # Aqui pegamos explicitamente os Tired/Falhas e transformamos em Inf
                    perf_matrix[i, j] = Inf
                end
            else
                # Se o método nem rodou nessa seed (dado faltante), fica Inf
                perf_matrix[i, j] = Inf
            end
        end
    end

    return perf_matrix, methods
end

"""
    generate_performance_profiles(df; title_suffix="")

Generate performance profiles for Time and Iterations.

Uses BenchmarkProfiles.jl signature:
    performance_profile(backend, matrix, solver_names; kwargs...)
    
where solver_names is a vector of strings with method names.
"""
function generate_performance_profiles(df::DataFrame)
    # --- CONFIGURAÇÃO VISUAL PARA ARTIGOS (P&B Friendly) ---

    # 1. Definir estilos de linha distintos
    # Usamos reshape(..., 1, :) para garantir que o Plots aplique um estilo por série (método)
    # Ordem sugerida: Sólido, Tracejado, Pontilhado, Traço-Ponto, Traço-Ponto-Ponto
    styles_list = [:solid, :dash, :dot, :dashdot, :dashdotdot]

   

    # -------------------------------------------------------

    # Time profile
    T_matrix, methods = build_performance_matrices(df, :Time)

    p_time = nothing
    if T_matrix !== nothing
        # Garante que temos estilos suficientes ciclando a lista
        n_methods = length(methods)
        line_styles = styles_list[1:min(n_methods, length(styles_list))]
        # reshape(repeat(styles_list, outer=ceil(Int, n_methods / length(styles_list)))[1:n_methods], 1, n_methods)

        p_time = performance_profile(
            PGFPlotsXBackend(),
            T_matrix,
            collect(methods),
            # logscale=true,           # Escala logarítmica
            ticks = :native,
            # xticks=custom_xticks,    # Aplica nossos ticks limpos
            linestyle=line_styles,   # Aplica os estilos de linha
            title="Wall-clock Time",
            legend=:bottomright,
            ylabel="Proportion of problems",
            xlabel="Performance ratio (τ)",
        )
    end

    # Iteration profile  
    I_matrix, methods_iter = build_performance_matrices(df, :Iterations)

    p_iter = nothing
    if I_matrix !== nothing
        n_methods_iter = length(methods_iter)
        line_styles_iter = reshape(repeat(styles_list, outer=ceil(Int, n_methods_iter / length(styles_list)))[1:n_methods_iter], 1, n_methods_iter)

        p_iter = performance_profile(
            PGFPlotsXBackend(),
            I_matrix,
            collect(methods_iter),
            ticks = :native,
            # logscale=true,
            # xscale=:log2,            # Força base 2
            # xticks=custom_xticks,    # Ticks limpos
            linestyle=line_styles,
            title="Iterations",
            legend=:bottomright,
            ylabel="Proportion of problems",
            xlabel="Performance ratio (τ)"
        )
    end

    return p_time, p_iter
end

#=============================================================================
    MAIN FUNCTIONS
=============================================================================#

"""
    generate_and_save_profiles(df, example_label, scenario_label; save_plots)

Generate performance profiles and save them as PDF.
"""
function generate_and_save_profiles(df::DataFrame,
    example_label::String,
    scenario_label::String;
    save_plots::Bool=true)

    # Generate profiles
    p_time, p_iter = generate_performance_profiles(df)

    # Save if requested
    if save_plots
        save_dir = plotsdir()
        mkpath(save_dir)

        if p_time !== nothing
            filename = "profile_$(example_label)_$(scenario_label)_time.pdf"
            filepath = joinpath(save_dir, filename)
            savefig(p_time, filepath)
            println("  ✓ Saved: $filename")
        end

        if p_iter !== nothing
            filename = "profile_$(example_label)_$(scenario_label)_iter.pdf"
            filepath = joinpath(save_dir, filename)
            savefig(p_iter, filepath)
            println("  ✓ Saved: $filename")
        end
    end

    return p_time, p_iter
end

"""
    process_example(example_num::String; save_plots=true)

Process one example: load data, generate profiles for A and B+C.

# Arguments
- `example_num`: "5.1", "5.2", or "5.3"
- `save_plots`: Save PDF files to plots directory (default: true)

# Returns
Dictionary with generated plots
"""
function process_example(example_num::String; save_plots::Bool=true)
    println("\n" * "="^70)
    println("Processing Example $example_num")
    println("="^70)

    example_label = replace(example_num, "." => "_")
    plots = Dict{String,Any}()

    #=========================================================================
        SCENARIO A (all methods)
    =========================================================================#

    println("\n[1/2] Scenario A (all methods)...")
    csv_A = datadir("sims", "results_$(example_label)A.csv")

    if isfile(csv_A)
        df_A = CSV.read(csv_A, DataFrame)
        println("  Loaded $(nrow(df_A)) records")

        p_time_A, p_iter_A = generate_and_save_profiles(
            df_A, example_label, "A";
            save_plots=save_plots
        )

        plots["A_time"] = p_time_A
        plots["A_iter"] = p_iter_A
    else
        @warn "File not found: $csv_A"
    end

    #=========================================================================
        SCENARIOS B+C (approximate methods only)
    =========================================================================#

    println("\n[2/2] Scenarios B+C (approximate methods only)...")

    csv_B = datadir("sims", "results_$(example_label)B.csv")
    csv_C = datadir("sims", "results_$(example_label)C.csv")

    # Combine B and C
    df_BC = DataFrame()
    for (label, csv_file) in [("B", csv_B), ("C", csv_C)]
        if isfile(csv_file)
            df = CSV.read(csv_file, DataFrame)
            append!(df_BC, df)
            println("  Loaded Scenario $label: $(nrow(df)) records")
        end
    end

    if nrow(df_BC) > 0
        # Filter to approximate methods only
        df_BC_approx = filter(r -> r.Method in METHODS_APPROX_ONLY, df_BC)
        println("  Filtered to approximate methods: $(nrow(df_BC_approx)) records")

        p_time_BC, p_iter_BC = generate_and_save_profiles(
            df_BC_approx, example_label, "BC";
            save_plots=save_plots
        )

        plots["BC_time"] = p_time_BC
        plots["BC_iter"] = p_iter_BC
    else
        @warn "No data for Scenarios B+C"
    end

    println("\n" * "="^70)
    println("Completed Example $example_num - Generated $(length(plots)) plots")
    println("="^70)

    return plots
end

"""
    process_all_examples(; save_plots=true)

Process all examples (5.1, 5.2, 5.3).

# Arguments  
- `save_plots`: Save PDF files to plots directory (default: true)

# Returns
Dictionary with all generated plots
"""
function process_all_examples(; save_plots::Bool=true)
    println("\n" * "="^70)
    println("PROCESSING ALL EXAMPLES")
    println("="^70)

    all_plots = Dict{String,Dict{String,Any}}()

    for example in ["5.1", "5.2", "5.3"]
        plots = process_example(example; save_plots=save_plots)
        all_plots[example] = plots
    end

    println("\n" * "="^70)
    println("✓ ALL EXAMPLES COMPLETED")
    println("="^70)

    if save_plots
        println("\nPDF files saved to: $(plotsdir())")
        println("Naming convention: profile_{example}_{scenario}_{metric}.pdf")
        println("  where example ∈ {5_1, 5_2, 5_3}")
        println("        scenario ∈ {A, BC}")
        println("        metric ∈ {time, iter}")
    end

    total_plots = sum(length(v) for v in values(all_plots))
    println("\nGenerated $total_plots total plots")

    return all_plots
end

#=============================================================================
    MAIN EXECUTION
=============================================================================#

# Uncomment to run automatically:
process_all_examples(save_plots=true)
# process_example("5.1", save_plots=true)

println("""
Performance Profile Generator Loaded
=====================================

Usage:
  process_example("5.1")           # Process Example 5.1
  process_all_examples()            # Process all examples

Output: PDF files in $(plotsdir())

Note: Non-converged instances marked as Inf (shown as failures in profile)
""")