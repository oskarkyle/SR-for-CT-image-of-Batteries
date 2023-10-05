#!/bin/bash


/Users/haoruilong/miniforge3/bin/python3 $1.py 
/Library/TeX/texbin/pdflatex $1.tex

rm *.aux *.log *.vscodeLog
rm *.tex

if [[ "$OSTYPE" == "darwin"* ]]; then
    open $1.pdf
else
    xdg-open $1.pdf
fi
