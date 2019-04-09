#!/bin/bash


echo "Run assignment 1: Erik Osinga"

echo "Creating the plotting directory if it does not exist"
if [ ! -d "plots" ]; then
  echo "Directory does not exist create it!"
  mkdir plots
fi

# Run scripts and pipe output to file and terminal
python3 question1.py 2>&1 | tee q1output.txt
python3 question2.py 2>&1 | tee q2output.txt
python3 question3.py 2>&1 | tee q3output.txt

echo ""
echo "Generating the pdf"
pdflatex main.tex
bibtex main.aux
pdflatex main.tex
pdflatex main.tex
