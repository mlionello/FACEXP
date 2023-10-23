from pathlib import Path
from facexp_models.RAVNESS_analyser import run_analyse

from multiprocessing import Pool, TimeoutError
with Pool(processes=8) as pool:
    for tr_ch in [0, 1]:
        for tr_rep in [0]:
            for tst_rep in [0]:
                for tst_intensity in [0, 1, 2]:
                    for tr_intensity in [0, 1, 2]:
                        outputid = f"trch_{tr_ch}_trrep_{tr_rep}_trintensity_{tr_intensity}_tstrep_{tst_rep}_tstintensity_{tst_intensity}"
                        inputdir = Path("/home/matteo.lionello/FACEXP/mediapipe/RAVDESS/")
                        outputid = inputdir / outputid
                        custom_cond = {
                            "tr_intensity": tr_intensity,
                            "tr_ch": tr_ch,
                            "tr_rep": tr_rep,
                            "tst_intensity": tst_intensity,
                            "tst_ch": 1,
                            "tst_rep": tst_rep,
                        }

                        run_analyse(inputdir, outputid, custom_cond, k_nn=1)
