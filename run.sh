#!/bin/bash

# Checking number of arguments
if [ "$#" -lt 1 ]; then
    echo "Invalid Arguments"
    exit 1
fi

if [ "$1" -eq "1" ]; then
    if [ "$#" -lt 5 ]; then
        echo "Invalid Arguments"
        exit 1
    fi
    python3 Q1/main.py $2 $3 $4 $5
    exit 0
elif [ "$1" -eq "2" ]; then
    if [ "$#" -lt 4 ]; then
        echo "Invalid Arguments"
        exit 1
    fi
    python3 Q2/main.py $2 $3 $4
    exit 0
elif [ "$1" -eq "3" ]; then
    if [ "$#" -lt 3 ]; then
        echo "Invalid Arguments"
        exit 1
    fi
    python3 Q3/main.py $2 $3
    exit 0
elif [ "$1" -eq "4" ]; then
    if [ "$#" -lt 4 ]; then
        echo "Invalid Arguments"
        exit 1
    fi
    python3 Q4/main.py $2 $3 $4
    exit 0
fi

echo "Invalid Arguments"
    exit 1