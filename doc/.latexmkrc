$pdf_mode = 5;  # Use XeLaTeX
$pdflatex = 'xelatex -synctex=1 -interaction=nonstopmode -file-line-error -output-directory=build %O %S';
$bibtex_use = 2;  # Use bibtex for bibliography

# Build directory
$out_dir = 'build';
$aux_dir = 'build';

# Default file (relative to doc/ directory)
@default_files = ('src/Project_Plan_and_Review.tex');

# Set BIBINPUTS to find bibliography files
$ENV{'BIBINPUTS'} = 'bib:' . ($ENV{'BIBINPUTS'} || '');

