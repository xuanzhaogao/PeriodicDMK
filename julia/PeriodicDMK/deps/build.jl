## Build script for PeriodicDMK shared library.
## Runs cmake to build libpdmk_jl.so from the C++ source.

const REPO_ROOT = abspath(joinpath(@__DIR__, "..", "..", ".."))
const CPP_DIR = joinpath(REPO_ROOT, "cpp")
const BUILD_DIR = joinpath(@__DIR__, "build")
const DEPS_FILE = joinpath(@__DIR__, "deps.jl")

function build()
    mkpath(BUILD_DIR)

    # Configure
    cmake_args = [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DPDMK_ENABLE_FLOAT=1",
    ]

    run(`cmake $(CPP_DIR) -B $(BUILD_DIR) $(cmake_args)`)

    # Build just the shared library target
    run(`cmake --build $(BUILD_DIR) --target pdmk_jl -- -j $(Sys.CPU_THREADS)`)

    # Find the built library
    libname = "libpdmk_jl.$(Sys.islinux() ? "so" : Sys.isapple() ? "dylib" : "dll")"
    libpath = joinpath(BUILD_DIR, libname)
    if !isfile(libpath)
        error("Build succeeded but $libpath not found")
    end

    # Write deps.jl so the package knows where the library is
    open(DEPS_FILE, "w") do f
        println(f, "const libpdmk_jl = \"$(escape_string(libpath))\"")
    end

    @info "PeriodicDMK built successfully" libpath
end

build()
