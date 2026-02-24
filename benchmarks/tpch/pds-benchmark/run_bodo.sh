#!/bin/sh

time {
    for i in {1..22}
    do
        mpiexec -n 1 python -m queries.bodo.q${i}
        sleep 2
    done
}
