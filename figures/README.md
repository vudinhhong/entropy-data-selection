# Figures and S1 Table

`fig1.pdf` to `fig4.pdf` are the figures the paper uses, written here by

    python ../code/07_figures/make_figures.py        # fig1
    python ../code/07_figures/make_pilot_figures.py  # fig2, fig3, fig4

from `../data/per_run_accuracies.csv`. Figs 2 to 4 go through the same two
modules that produce the corresponding tables, `16_contrast` and
`18_direction`, so a figure cannot disagree with the table beside it.
`make_figures.py --budget` draws one more curve that the paper reports in the
text and does not plot. Re-running the script overwrites it,
which is the point: the figure and the tables are two renderings of one set
of numbers and cannot drift apart. If the data are missing the script stops
with an error rather than writing a blank figure.

`S1_Table.tex` and `S1_Table.pdf` are the supporting table, shipped built
because `code/11_s1_table/make_s1.py` needs per-item correctness vectors that
the archive does not carry.

The manuscript is not included in this archive. See the related identifier
in the Zenodo record for the paper itself.
