#!/bin/sh

time {
    for i in {1..22}
    do
        python -m queries.bodo.q${i}
        sleep 2
    done
}
