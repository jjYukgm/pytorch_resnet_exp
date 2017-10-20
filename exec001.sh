#!/bin/bash

python main.py -n r_110 -c e200 -t --ed && \
python main.py -n r_37 --dn r_110_ed --r3 --pn r_37 -c n_e200 &&\
python main.py -n r_37_r3 --dn r_110_ed -t -c e200 &&\
python main.py -n r_73 --dn r_110_ed --r3 --pn r_73 -c n_e200 &&\
python main.py -n r_73_r3 --dn r_110_ed -t -c e200 &&\


python main.py -n r_91 &&\
python main.py -n r_91 -t -c e200 &&\

python main.py -n r_91 --dn r_110_ed --r3 --pn r_91 -c e200 &&\
python main.py -n r_91_r3 --dn r_110_ed -t -c e200 &&\


