"""Dispositional traits (Methods, "Dispositional traits"; Table "disp";
Supplementary Note "Dispositional cells").

The animal of the main grid is replaced by a disposition in how an assistant
answers (rude-polite, harsh-gentle, persuasive-neutral). Everything else is the
main grid's SP pipeline; this package holds only what differs:

* ``judges``   -- the SP judge sentences (training sentence and the eight
                  wordings used for emission).
* ``items``    -- preference-evaluation items about the student's own manner and
                  the answer-evaluation templates (data in configs/dispositions/).
* ``carriers`` -- the first-token table that lets the syllable carrier's name
                  exclusion cover the adjectives.
* ``analysis`` -- the statistics behind disp_grid.json, disp_endpoint.json,
                  disp_lambda.json and disp_screen.json.
"""
