import sys

""""
This file writes latex figures to copy paste into the .tex file
It uses minipages to put two figures side by side
"""


# Two figures look like this:
"""
\begin{figure}[ht]
\centering
\begin{minipage}[t]{.5\textwidth}
  \centering
  \includegraphics[width=1.0\linewidth]{./plots/q1d2.png}
  \captionsetup{width=0.8\linewidth}
  \captionof{figure}{..}
  \label{fig:fig9}
\end{minipage}%
\begin{minipage}[t]{.5\textwidth}
  \centering
  \includegraphics[width=1.0\linewidth]{./plots/q1d3.png}
  \captionsetup{width=0.8\linewidth}
  \captionof{figure}{..}
  \label{fig:fig10}
\end{minipage}
\end{figure}
"""


def write_latex_figures(mainq, subq, fignumstart, itstring, it, numit=1,caption='..'):
	"""
	Write latex figures from main question and sub question
	Figures are numbered starting from fignumstart
	No caption is defined yet
	
	itstring is '_' or ''
	it is the figure to start from (e.g., q5a_{it} for q5a_1.png)
	numit is the amount of figures (itend-it+1)
	"""

	with open(f'./fig{mainq}_{subq}.txt','w') as file:
		i = it
		while i < (it+numit):
			file.write(r'\begin{figure}[ht]')
			file.write(r'\centering')
			file.write('\n')
			file.write(r'\begin{minipage}[t]{.5\textwidth}')
			file.write('\n')
			file.write(r'\centering')
			file.write('\n')
			file.write(r'\includegraphics[width=1.0\linewidth]{'+f'./plots/q{mainq}{subq}{itstring}{i}.png'+'}')
			i += 1
			file.write('\n')
			file.write(r'\captionsetup{width=0.8\linewidth}')
			file.write('\n')
			file.write(r'\captionof{figure}{'+f'{caption}'+'}')
			file.write('\n')
			file.write(r'\label{'+f'fig:fig{fignumstart}'+'}')
			fignumstart +=1
			file.write('\n')
			file.write(r'\end{minipage}%')
			file.write('\n')

			# Second figure in same minipage
			file.write(r'\begin{minipage}[t]{.5\textwidth}')
			file.write('\n')
			file.write(r'\centering')
			file.write('\n')
			if i < (it+numit):
				file.write(r'\includegraphics[width=1.0\linewidth]{'+f'./plots/q{mainq}{subq}{itstring}{i}.png'+'}')
				i += 1
				file.write('\n')
				file.write(r'\captionsetup{width=0.8\linewidth}')
				file.write('\n')
				file.write(r'\captionof{figure}{'+f'{caption}'+'}')
				file.write('\n')
				file.write(r'\label{'+f'fig:fig{fignumstart}'+'}')
				fignumstart+=1
				file.write('\n')
			file.write(r'\end{minipage}%')
			file.write('\n')
			file.write(r'\end{figure}')

			file.write('\n')
			file.write('\n')

mainq, subq, fignumstart, itstring, it, numit = sys.argv[1:]
if itstring == 'None':
	itstring = ''
else:
	itstring = '_'

if subq == 'None':
	subq = ''

fignumstart = int(fignumstart)
it = int(it)
numit = int(numit)

print (f"Main question {mainq}, subquestion {subq}. Starting at Figure {it} with label fig{fignumstart}")
print ("First figure:")
print (f"./plots/q{mainq}{subq}{itstring}{it}.png")
print ("Last figure:")
print (f"./plots/q{mainq}{subq}{itstring}{it+numit-1}.png")


# caption = 'Result of the two sample tests on the datastream indicated above the figure.'
# caption = 'Density field with power spectrum defined as indicated above the figure.'
# caption = 'Numerical solution to ODE case .'
# caption = 'The $y$-position of the first ten particles in the $y$-direction as a function of the scale factor $a$.'
# caption = '$x$--$y$ slice of the 16x16x16 NGP mass grid for the z-value indicated above the plot. The color denotes the mass in a cell'
# caption = 'Mass of the indicated cell as a function of the $x$-position of a single particle. The mass is calculated with the NGP method.'
# caption = 'Mass of the indicated cell as a function of the $x$-position of a single particle. The mass is calculated with the CIC method.'
# caption = '$x$--$y$ slice of the 16x16x16 CIC mass grid for the z-value indicated above the plot. The color denotes the mass in a cell'
# caption = 'Discrete Fourier transform of $2sin(2\pi x)$, sampled at 64 values between 0 and $6\pi$. The analytical result is plotted as a dashed line. The FFT implemented in this work overlaps perfectly with the Numpy FFT'
# caption = 'Discrete Fourier transform of '
# caption = 'Centered slice for the $X$--$X$ plane of the potential $\Phi(r)$ indicated by the colorbar.'
# caption = '$x$--$y$ slice of the 16x16x16 CIC Potential $\Phi(r)$ grid for the z-value indicated above the plot. The color denotes the potential of a cell'

caption = '..'
caption = 'Quad tree of the 1200 points. The leaf nodes are shown as boxes that contain at most 12 points.'

write_latex_figures(mainq, subq, fignumstart, itstring, it, numit, caption)

print (f"File written: ./fig{mainq}_{subq}.txt ")


"""
COMMANDS TO GENERATE ALL FIGURES STARTING FROM 1e:
There is a small error in the assigning of the figure number (fig10), but we fix this manually in the pdf


python3 write_latex_figures.py 1 e 10 None 0 10
python3 write_latex_figures.py 2 None 21 _ 0 3
python3 write_latex_figures.py 3 None 24 _ 0 3
python3 write_latex_figures.py 4 c 27 None 1 2
python3 write_latex_figures.py 4 d 29 None 1 2
python3 write_latex_figures.py 5 a 31 _ 0 4
python3 write_latex_figures.py 5 b 35 None 1 2
python3 write_latex_figures.py 5 c 37 None 1 2
python3 write_latex_figures.py 5 c 39 _ 0 4
python3 write_latex_figures.py 5 d 43 None 1 1
python3 write_latex_figures.py 5 e 44 None 1 4
python3 write_latex_figures.py 5 f 48 None 1 2
python3 write_latex_figures.py 5 f 50 _ 0 4
# This one does need some editing of the figure names in .tex 
python3 write_latex_figures.py 6 None 54 _ 0 6 
python3 write_latex_figures.py 7 None 60 _ 1 1 


"""