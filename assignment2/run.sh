#!/bin/bash


echo "Run assignment 1: Erik Osinga"

echo "Creating the plotting directory if it does not exist"
if [ ! -d "plots" ]; then
  echo "Directory does not exist create it!"
  mkdir plots
fi
if [ ! -d "plots/movie" ]; then
  mkdir movie
fi

# Run scripts and pipe output to file and terminal
#python3 question1.py 2>&1 | tee q1output.txt
#python3 question2.py 2>&1 | tee q2output.txt
#python3 question3.py 2>&1 | tee q3output.txt

# Create movie of the frames produced by question4.py
ffmpeg -framerate 30 -pattern_type glob -i "plots/movie/4c_*.png" -s:v 640x480 -c:v libx264 -profile:v high -level 4.0 -crf 10 -tune animation -preset slow -pix_fmt yuv420p -r 25 -threads 0 -f mp4 quest4c.mp4




echo ""
echo "Generating the pdf"
pdflatex main.tex
bibtex main.aux
pdflatex main.tex
pdflatex main.tex
