#!/bin/tcsh
# Run from textbook/. Build both the integrated text and the diagram review PDF.
foreach target (main tikz_figures)
    foreach pass (1 2 3)
        xelatex -interaction=nonstopmode -halt-on-error "$target.tex"
        if ($status != 0) exit $status
    end
end
exit 0
