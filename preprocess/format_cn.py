import numpy as np
import pandas as pd
from utils import *


def format_cn_profile(
    seg_ucn: str,
    bbc_ucn: str,
    out_seg_file: str,
    out_seg_bed_file: str,
    out_seg_acopy_file: str,
    out_seg_bcopy_file: str,
    out_seg_cbaf_file: str,
    exclude_baf_eps=1e-6,
    exclude_baf_tol=2e-2,
    exclude_seg_len=5e6,
    laplace_alpha=0.01,
):
    """
    return:
    0-based
    1. segments dataframe, #CHR,START,END,mhBAF,BBC_mhBAF
    2. segment by clone Acopy dataframe
    3. segment by clone Bcopy dataframe
    4. segment by clone BAF dataframe

    filter segments with following criteria
    1. short than exclude_seg_len
    2. normal-like, exclude_baf_eps
    3. expected BAF and BBC-ave BAF deviated more than exclude_baf_tol
    """
    segs, clones = read_seg_ucn_file(seg_ucn)
    n_clones = len(clones)
    chs = segs["#CHR"].unique().tolist()
    segs.loc[:, "exclude"] = 0
    segs.loc[:, "mhBAF"] = 0.0
    segs.loc[:, "BBC_mhBAF"] = 0.0
    bbcs = read_bbc_ucn_file(bbc_ucn)

    segs_acopies = []
    segs_bcopies = []
    segs_clone_bafs = []

    # convert to 0-based indexing
    for ch in chs:
        segs_ch = segs.loc[segs["#CHR"] == ch]
        segs.loc[segs_ch.index[0], "START"] -= 1
        bbcs_ch = bbcs.loc[bbcs["#CHR"] == ch]
        bbcs.loc[bbcs_ch.index[0], "START"] -= 1

        for i in range(len(segs_ch) - 1):
            idx = segs_ch.index[i]
            jdx = segs_ch.index[i + 1]
            if segs_ch.loc[idx, "END"] != segs_ch.loc[jdx, "START"]:
                segs.loc[jdx, "START"] -= 1
        for i in range(len(bbcs_ch) - 1):
            idx = bbcs_ch.index[i]
            jdx = bbcs_ch.index[i + 1]
            if bbcs_ch.loc[idx, "END"] != bbcs_ch.loc[jdx, "START"]:
                bbcs.loc[jdx, "START"] -= 1

        for sidx in segs_ch.index:
            row = segs_ch.loc[sidx]
            seg_start = row["START"]
            seg_end = row["END"]
            seg_len = seg_end - seg_start

            seg_bafs_pbbc = bbcs_ch.loc[
                (bbcs_ch["START"] >= seg_start) & (bbcs_ch["END"] <= seg_end), "BAF"
            ].to_numpy()
            bbc_mhbaf = np.mean(seg_bafs_pbbc)

            acopies = np.zeros(n_clones, dtype=np.int32)
            bcopies = np.zeros(n_clones, dtype=np.int32)
            purities = np.zeros(n_clones, dtype=np.float128)
            for j, clone in enumerate(clones):
                acopies[j] = int(row[f"cn_{clone}"].split("|")[0])
                bcopies[j] = int(row[f"cn_{clone}"].split("|")[1])
                purities[j] = float(row[f"u_{clone}"])
            ccopies = acopies + bcopies
            bafs = np.divide(
                bcopies + laplace_alpha,
                ccopies + laplace_alpha * 2,
                out=np.zeros_like(ccopies, dtype=np.float32),
                where=(ccopies != 0),
            )
            afrac = np.dot(acopies, purities)
            bfrac = np.dot(bcopies, purities)
            mhbaf = bfrac / (afrac + bfrac)
            if np.all(acopies == 1) and np.all(bcopies == 1):
                segs.loc[sidx, "exclude"] = 1
            if np.all(acopies[1:] == 2) and np.all(bcopies[1:] == 2):
                segs.loc[sidx, "exclude"] = 1
            if abs(mhbaf - 0.5) <= exclude_baf_eps:
                segs.loc[sidx, "exclude"] = 1

            if abs(bbc_mhbaf - mhbaf) > exclude_baf_tol:
                segs.loc[sidx, "exclude"] = 1
            if seg_len < exclude_seg_len:
                segs.loc[sidx, "exclude"] = 1

            if segs.loc[sidx, "exclude"] != 1:
                segs.loc[sidx, "mhBAF"] = mhbaf
                segs.loc[sidx, "BBC_mhBAF"] = bbc_mhbaf
                segs_acopies.append(acopies)
                segs_bcopies.append(bcopies)
                segs_clone_bafs.append(bafs)

    num_incl_segs = segs["exclude"].sum()
    print(f"#informative-segments={len(segs) - num_incl_segs}/{len(segs)}")

    # save block range information
    segs = segs[segs["exclude"] == 0]
    segs.to_csv(
        out_seg_file,
        index=False,
        header=True,
        sep="\t",
        columns=["#CHR", "START", "END", "mhBAF", "BBC_mhBAF"],
    )

    segs.to_csv(
        out_seg_bed_file,
        index=False,
        header=False,
        sep="\t",
        columns=["#CHR", "START", "END"],
    )

    # save copy-state information
    pd.DataFrame(segs_acopies, columns=clones).to_csv(
        out_seg_acopy_file, index=False, header=True, sep="\t"
    )
    pd.DataFrame(segs_bcopies, columns=clones).to_csv(
        out_seg_bcopy_file, index=False, header=True, sep="\t"
    )
    pd.DataFrame(segs_clone_bafs, columns=clones).to_csv(
        out_seg_cbaf_file, index=False, header=True, sep="\t"
    )
    return
