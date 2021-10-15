#!/bin/bash
#PBS -q qexp
#PBS -N pymoo
#PBS -l select=1:ncpus=36:mpiprocs=36,walltime=1:00:00
#PBS -J 1-3
#PBS -A OPEN-17-39

# Fronta qexp (nastavena vyse parametrem -q) neodecita hodiny z projektu,
# ma pro spusteni nejvyssi prioritu, ale umoznuje walltime max. 1hod
# a najednou muzete mit prideleny max. 4 uzly (v dokumentaci 8, asi to zmenili).
# Pro narocnejsi vypocty specifikujte frontu qprod a walltime nastavte dle
# potreby az na 12 hod.

# Pro anselm: 16-jadrove uzly, -J 1-6 (celkem spusti 96 behu z kazdeho PARAMx).
# Pro salomon na mistech ncpus a mpiprocs zapiste 24 a -J 1-4 pro cca 100 (presne 96) behu.
# Pro barbora na mistech ncpus a mpiprocs zapiste 36 a -J 1-3 pro cca 100 (presne 108) behu.
# Pro karolina

# Pojmenovani jobu (parametr -N) zmente podle libosti (a stejne je nutno take
# nastavit promenou PROJECT v Makefile!!).
# Pokud naspoustite spoustu jobu, ale zjistite, ze jste nekde udelal chybu
# a potrebujete je vsechny ukoncit, provedete to spustenim skriptu qdel.sh
# v podadresari src1.

ml OpenMPI
ml Python/3.9.5-GCCcore-10.3.0

cd $PBS_O_WORKDIR

mpiexec python $@
