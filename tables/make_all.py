"""Write every table body of the paper (the files included as tabs/<name>.tex) from data/.

    python tables/make_all.py --out <directory> [--data data]

Table 1 (dispositional traits) -> disp.tex; Extended Data Tables 1-7 -> main.tex,
judge_movement.tex, further_pairs.tex, chan.tex, alpha.tex, time.tex, judge.tex.
"""
import common as C
import table_alpha
import table_channel
import table_dispositions
import table_further
import table_judges
import table_main
import table_judge_movement
import table_time

TABLES = [("disp", table_dispositions), ("main", table_main), ("judge_movement", table_judge_movement), ("further_pairs", table_further),
          ("chan", table_channel), ("alpha", table_alpha), ("time", table_time), ("judge", table_judges)]

if __name__ == "__main__":
    a = C.cli("Write all table bodies.")
    for name, mod in TABLES:
        print(C.write(a.out, name, mod.build(a.data)))
