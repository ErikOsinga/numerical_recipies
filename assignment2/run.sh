#!/bin/bash


echo "Run assignment 1: Erik Osinga"

echo "Creating the plotting directory if it does not exist"
if [ ! -d "plots" ]; then
  echo "Directory does not exist create it!"
  mkdir plots
fi
if [ ! -d "plots/movie" ]; then
  mkdir plots/movie
fi

echo "Download data for exercise 1"
if [ ! -e randomnumbers.txt ]; then
  wget strw.leidenuniv.nl/~nobels/coursedata/randomnumbers.txt
fi


# Run scripts and pipe output to file and terminal
#python3 question1.py 2>&1 | tee q1output.txt
#python3 question2.py 2>&1 | tee q2output.txt
#python3 question3.py 2>&1 | tee q3output.txt

# Create movie of the frames produced by question4.py
ffmpeg -framerate 30 -pattern_type glob -i "plots/movie/4c_*.png" -s:v 640x480 -c:v libx264 -profile:v high -level 4.0 -crf 10 -tune animation -preset slow -pix_fmt yuv420p -r 25 -threads 0 -f mp4 quest4c.mp4

# The xy direction
ffmpeg -framerate 30 -pattern_type glob -i "plots/movie/4d_xy*.png" -s:v 640x480 -c:v libx264 -profile:v high -level 4.0 -crf 10 -tune animation -preset slow -pix_fmt yuv420p -r 25 -threads 0 -f mp4 quest4dxy.mp4

# The xz direction
ffmpeg -framerate 30 -pattern_type glob -i "plots/movie/4d_xz*.png" -s:v 640x480 -c:v libx264 -profile:v high -level 4.0 -crf 10 -tune animation -preset slow -pix_fmt yuv420p -r 25 -threads 0 -f mp4 quest4dxz.mp4

# The yz direction
ffmpeg -framerate 30 -pattern_type glob -i "plots/movie/4d_yz*.png" -s:v 640x480 -c:v libx264 -profile:v high -level 4.0 -crf 10 -tune animation -preset slow -pix_fmt yuv420p -r 25 -threads 0 -f mp4 quest4dyz.mp4


echo "Download data for exercise 6"
if [ ! -e GRBs.txt ]; then
  wget strw.leidenuniv.nl/~nobels/coursedata/GRBs.txt
fi

echo "Download data for exercise 7"
if [ ! -e colliding.hdf5 ]; then
  wget strw.leidenuniv.nl/~nobels/coursedata/colliding.hdf5
fi



echo ""
echo "Generating the pdf"
pdflatex main.tex
bibtex main.aux
pdflatex main.tex
pdflatex main.tex
