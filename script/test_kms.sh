#!/bin/bash

# python2 kmain.py -n r_37_nr1.0E-01 -t --kmean &&\
python2 kmain.py -n r_37_nr1.0E-01 -t --kmean -c e200 &&\

python2 ./analysis.py --km -n r_37_nr1.0E-01_best,
