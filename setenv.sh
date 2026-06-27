module -q reset
module load modules/2.4
module use ~/modules
module -q load python gcc/14 cmake openmpi gdb openblas/single flexiblas llvm/19 doxygen papi perf-linux
