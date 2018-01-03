#!/bin/bash

# python2 kmain.py -n r_37_nr1.0E-01 -t --kmean &&\

# python2 kmain.py -n r_37_nr1.0E-01 -t --kmean -c e200 &&\
# python2 kmain.py -n r_37_nr1.5E-01 -t --kmean &&\
# python2 kmain.py -n r_37_nr1.5E-01 -t --kmean -c e200 &&\
# python2 kmain.py -n r_37_nr2.0E-01 -t --kmean &&\
# python2 kmain.py -n r_37_nr2.0E-01 -t --kmean -c e200 &&\
# python2 ./analysis.py --km -n r_37_nr1.0E-01_best,r_37_nr1.0E-01_e200,\
# r_37_nr1.5E-01_best,r_37_nr1.5E-01_e200,\
# r_37_nr2.0E-01_best,r_37_nr2.0E-01_e200 > km_ana_log.txt \
# || echo "Error terminal analysis" && exit

# python2 kmain.py -n r_37d3_nr1.5E-01 -t --kmean &&\
# python2 kmain.py -n r_37d3_nr1.5E-01 -t --kmean -c e200 &&\

python2 kmain.py -n r_37d3_nr1.0E-01 -t --kmean &&\
python2 kmain.py -n r_37d3_nr1.0E-01 -t --kmean -c e200 &&\
python2 kmain.py -n r_37d3_nr2.0E-01 -t --kmean &&\
python2 kmain.py -n r_37d3_nr2.0E-01 -t --kmean -c e200 &&\
python2 ./analysis.py --km -n r_37d3_nr1.0E-01_best,r_37d3_nr1.0E-01_e200,\
r_37d3_nr2.0E-01_best,r_37d3_nr2.0E-01_e200 > km_ana_log2.txt \
|| echo "Error terminal analysis uc" && exit
# r_37d3_nr1.5E-01_best,r_37d3_nr1.5E-01_e200,\