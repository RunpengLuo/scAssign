import os
import sys
from collections import OrderedDict
import gzip
import pandas as pd
import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix

import subprocess
from io import StringIO


def get_ord2chr(ch="chr"):
    return [f"{ch}{i}" for i in range(1, 23)] + [f"{ch}X", f"{ch}Y"]


def get_chr2ord(ch):
    chr2ord = {}
    for i in range(1, 23):
        chr2ord[f"{ch}{i}"] = i
    chr2ord[f"{ch}X"] = 23
    chr2ord[f"{ch}Y"] = 24
    return chr2ord


def count_comment_lines(filename: str, comment_symbol="#"):
    num_header = 0
    if filename.endswith(".gz"):
        fd = gzip.open(filename, "rt")
    else:
        fd = open(filename, "r")
    for line in fd:
        if line.startswith(comment_symbol):
            num_header += 1
        else:
            break
    fd.close()
    return num_header


def sort_chroms(chromosomes: list):
    assert len(chromosomes) != 0
    ch = "chr" if str(chromosomes[0]).startswith("chr") else ""
    chr2ord = get_chr2ord(ch)
    return sorted(chromosomes, key=lambda x: chr2ord[x])


def read_cn_profile(seg_ucn: str):
    """
    get copy-number profile per clone from HATCHet seg file
    """
    seg_df = pd.read_table(seg_ucn)
    # all samples share same copy-number states for each segment, just different purity
    samples = seg_df["SAMPLE"].unique().tolist()
    groups_sample = seg_df.groupby("SAMPLE", sort=False)
    seg_df = groups_sample.get_group(samples[0])

    n_clones = sum(1 for c in seg_df.columns if str.startswith(c, "cn_clone")) + 1
    clones = ["normal"] + [f"clone{i}" for i in range(1, n_clones)]
    chs = sort_chroms(seg_df["#CHR"].unique().tolist())
    seg_df["#CHR"] = pd.Categorical(seg_df["#CHR"], categories=chs, ordered=True)
    seg_df.sort_values(by=["#CHR", "START"], inplace=True, ignore_index=True)

    groups_ch = seg_df.groupby(by="#CHR", sort=False)

    ch2segments = OrderedDict()  # ch -> position array
    ch2a_profile = OrderedDict()  # ch -> cn profile
    ch2b_profile = OrderedDict()  # ch -> cn profile
    for ch in chs:
        seg_df_ch = groups_ch.get_group(ch)
        num_segments_ch = len(seg_df_ch)
        ch2segments[ch] = seg_df_ch[["START", "END"]].to_numpy(dtype=np.int64)
        a_profile = np.zeros((num_segments_ch, n_clones), dtype=np.int8)
        b_profile = np.zeros((num_segments_ch, n_clones), dtype=np.int8)
        for j, clone in enumerate(clones):
            a_profile[:, j] = seg_df_ch.loc[:, f"cn_{clone}"].apply(
                func=lambda c: int(c.split("|")[0])
            )
            b_profile[:, j] = seg_df_ch.loc[:, f"cn_{clone}"].apply(
                func=lambda c: int(c.split("|")[1])
            )
        ch2a_profile[ch] = a_profile
        ch2b_profile[ch] = b_profile
    return chs, clones, ch2segments, ch2a_profile, ch2b_profile


def get_cn_probability(
    chs: list, ch2a_profile: dict, ch2b_profile: dict, laplace_alpha=0.01
):
    """
    compute laplace-smoothed copy-number probability
    """
    ch2probs = {}
    ch2masks = {}
    for ch in chs:
        a_profile = ch2a_profile[ch]
        b_profile = ch2b_profile[ch]
        c_profile = a_profile + b_profile
        cna_probabilities = np.divide(
            b_profile + laplace_alpha,
            c_profile + laplace_alpha * 2,
            out=np.zeros_like(c_profile, dtype=np.float32),
            where=(c_profile != 0),
        )
        ch2probs[ch] = cna_probabilities
        # mask a segment if all clones are copy-neutral
        ch2masks[ch] = np.all(a_profile == b_profile, axis=1)
    return ch2probs, ch2masks


def read_barcodes(bc_file: str):
    barcodes = []
    with open(bc_file, "r") as fd:
        for line in fd:
            barcodes.append(line.strip().split("\t")[0])
        fd.close()
    return barcodes


def read_VCF(vcf_file: str, has_phase=False):
    fields = "%CHROM\t%POS"
    names = ["#CHR", "POS"]
    format_tags = []
    if has_phase:
        format_tags.extend(["%GT", "%PS"])
        names.extend(["GT", "PS"])
    if len(format_tags) > 0:
        fields = fields + "\t[" + "\t".join(format_tags) + "]"
    fields += "\n"
    cmd = ["bcftools", "query", "-f", fields, vcf_file]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    snps = pd.read_csv(StringIO(result.stdout), sep="\t", header=None, names=names)
    if has_phase:
        # Drop entries without phasing output
        snps = snps[(snps["GT"] != ".") & (snps["PS"] != ".")]
        snps["GT"] = snps["GT"].apply(func=lambda v: v[0])  # USEREF
    assert not snps.duplicated().any(), f"{vcf_file} has duplicated rows"
    assert not snps.duplicated(subset=["#CHR", "POS"]).any(), (
        f"{vcf_file} has duplicated rows"
    )
    return snps


def read_VCF_cellsnp_err_header(vcf_file: str):
    """cellsnp-lite has issue with its header"""
    fields = "%CHROM\t%POS\t%INFO"
    names = ["#CHR", "POS", "INFO"]
    cmd = ["bcftools", "query", "-f", fields, vcf_file]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    snps = pd.read_csv(StringIO(result.stdout), sep="\t", header=None, names=names)

    def extract_info_field(info_str, key):
        for field in info_str.split(";"):
            if field.startswith(f"{key}="):
                return int(field.split("=")[1])
        raise ValueError("nan-DP field")

    snps["DP"] = snps["INFO"].apply(lambda x: extract_info_field(x, "DP"))
    snps = snps.drop(columns="INFO")
    return snps


def get_chr_sizes(sz_file: str):
    chr_sizes = OrderedDict()
    with open(sz_file, "r") as rfd:
        for line in rfd.readlines():
            ch, sizes = line.strip().split()
            chr_sizes[ch] = int(sizes)
        rfd.close()
    return chr_sizes


def read_baf_file(baf_file: str):
    baf_df = pd.read_table(
        baf_file,
        names=["#CHR", "POS", "SAMPLE", "REF", "ALT"],
        dtype={
            "#CHR": object,
            "POS": np.uint32,
            "SAMPLE": object,
            "REF": np.uint32,
            "ALT": np.uint32,
        },
    )
    return baf_df


def subset_baf(
    baf_df: pd.DataFrame, ch: str, start: int, end: int, is_last_block=False
):
    if ch != None:
        baf_ch = baf_df[baf_df["#CHR"] == ch]
    else:
        baf_ch = baf_df
    if baf_ch.index.name == "POS":
        pos = baf_ch.index
    else:
        pos = baf_ch["POS"]
    if is_last_block:
        return baf_ch[(pos >= start) & (pos <= end)]
    else:
        return baf_ch[(pos >= start) & (pos < end)]


def read_seg_ucn_file(seg_ucn_file: str):
    segs_df = pd.read_table(seg_ucn_file, sep="\t")
    chs = sort_chroms(segs_df["#CHR"].unique().tolist())
    segs_df["#CHR"] = pd.Categorical(segs_df["#CHR"], categories=chs, ordered=True)
    segs_df.sort_values(by=["#CHR", "START"], inplace=True, ignore_index=True)

    clones = [cname[3:] for cname in segs_df.columns if cname.startswith("cn_")]
    segs_df.loc[:, "CNP"] = segs_df.apply(
        func=lambda r: ";".join(r[f"cn_{c}"] for c in clones), axis=1
    )
    # segs_df.loc[:, "seg-length"] = segs_df["end"] - segs_df["start"]
    # segs_df.loc[:, "seg-position"] = segs_df.apply(
    #     func=lambda r: str(r["#CHR"]) + "_" + str(r["start"]) + "_" + str(r["end"]),
    #     axis=1,
    # )
    return segs_df, clones


def read_bbc_ucn_file(bbc_ucn_file: str):
    bbcs_df = pd.read_table(bbc_ucn_file, sep="\t")
    chs = sort_chroms(bbcs_df["#CHR"].unique().tolist())
    bbcs_df["#CHR"] = pd.Categorical(bbcs_df["#CHR"], categories=chs, ordered=True)
    bbcs_df.sort_values(by=["#CHR", "START"], inplace=True, ignore_index=True)
    # clones = [cname[3:] for cname in bbcs_df.columns if cname.startswith("cn_")]
    # bbcs_df["CNP"] = bbcs_df.apply(
    #     func=lambda r: ";".join(r[f"cn_{c}"] for c in clones), axis=1
    # )
    return bbcs_df


def BBC_segmentation(bbcs_df: pd.DataFrame):
    assert len(bbcs_df["SAMPLE"].unique()) == 1, ">1 samples"
    # segment BBC by chromosome and cluster ID
    group_name_to_indices = bbcs_df.groupby(
        (
            (bbcs_df["#CHR"] != bbcs_df["#CHR"].shift())
            | (bbcs_df["start"] != bbcs_df["end"].shift())
            | (bbcs_df["CLUSTER"] != bbcs_df["CLUSTER"].shift())
        ).cumsum(),
        # cumulative sum increments whenever a True is encountered, thus creating a series of monotonically
        # increasing values we can use as segment numbers
        sort=False,
    ).indices
    for group_name, indices in group_name_to_indices.items():
        bbcs_df.loc[indices, "segment"] = group_name

    aggregation_rules = {
        "#CHR": "first",
        "SAMPLE": "first",
        "start": "min",
        "end": "max",
        "#SNPS": "sum",
        "BAF": "mean",
        "RD": "mean",
        "COV": "mean",
        "ALPHA": "sum",
        "BETA": "sum",
        "CLUSTER": "first",
        "CNP": "first",
    }
    bbcs_df = bbcs_df.groupby(["segment", "SAMPLE"]).agg(aggregation_rules)
    return bbcs_df


def read_verify_file(verify_file: str):
    def simp_type(t: str):
        return t if "tumor" in t.lower() else "normal"

    def simp_type2(t: str):
        return "tumor" if "tumor" in t.lower() else "normal"

    # cell_id cell_types final_type
    verify_df = pd.read_table(verify_file)
    verify_df = verify_df.rename(columns={"cell_id": "BARCODE"})
    if "met_subcluster" in verify_df.columns.tolist():
        print("use column met_subcluster as final_type")
        verify_df["final_type"] = verify_df["met_subcluster"]

    if "final_type" not in verify_df.columns.tolist():
        assert "cell_types" in verify_df.columns.tolist(), (
            "cell_types column does not exist"
        )
        print("use column cell_types as final_type")
        verify_df["final_type"] = verify_df["cell_types"]
    if "simp_type" not in verify_df.columns.tolist():
        verify_df["simp_type"] = verify_df.apply(
            func=lambda r: simp_type(r["final_type"]), axis=1
        )
    if "simp_type2" not in verify_df.columns.tolist():
        verify_df["simp_type2"] = verify_df.apply(
            func=lambda r: simp_type2(r["final_type"]), axis=1
        )
    return verify_df


def mark_co_type(assign_df: pd.DataFrame, a_col="simp_type", b_col="Decision"):
    """
    add a column to indicate consistency between assignments and `solution`
    """

    def co_type(d: str, t: str):
        if t == "tumor" and (d.lower().startswith("clone") or d.lower() == "tumor"):
            return "tumor"
        if t == "normal" and d == "normal":
            return "normal"
        return f"{t}_{d}"

    assign_df.loc[:, "co_type"] = assign_df.apply(
        func=lambda r: co_type(r[b_col], r[a_col]), axis=1
    )
    return assign_df


def index_states(chs: list, ch2segments: dict, ch2a_profile: dict, ch2b_profile: dict):
    state2id = {}
    id2state = []
    ch2sids = {}
    pos2states = {}
    sid = 0
    for ch in chs:
        ch2sids[ch] = []
        for i in range(len(ch2a_profile[ch])):
            aa = ch2a_profile[ch][i]
            bb = ch2b_profile[ch][i]
            [seg_s, seg_t] = ch2segments[ch][i]
            state = ";".join(f"{aa[j]}|{bb[j]}" for j in range(len(aa)))
            if state not in state2id:
                state2id[state] = sid
                id2state.append(state)
                sid += 1
            ch2sids[ch].append(state2id[state])
            pos2states[(ch, seg_s, seg_t)] = state
    return state2id, id2state, ch2sids, pos2states


def read_hair_file(hair_file: str, nsnps: int):
    """
    read raw hair file from HapCUT2
    hair=one read per row, list of supported alleles
    n=#SNPs
    hair array: shape (n, 4), first row is dummy.
    """
    hairs = np.zeros((nsnps, 4), dtype=np.int16)
    mapper = {"00": 0, "01": 1, "10": 2, "11": 3}
    mapper_rev = {"00": 3, "01": 2, "10": 1, "11": 0}
    ctr = 0
    with open(hair_file, "r") as hair_fd:
        for line in hair_fd:
            ctr += 1
            if ctr % 100000 == 0:
                print(ctr)
            fields = line.strip().split(" ")[:-1]
            nblocks = int(fields[0])
            for i in range(nblocks):
                var_start = int(fields[2 + (i * 2)])  # 1-based
                phases = fields[2 + (i * 2 + 1)]
                var_end = var_start + len(phases) - 1

                pvar_idx = [mapper[phases[j : j + 2]] for j in range(len(phases) - 1)]
                pvar_idx_rev = [
                    mapper_rev[phases[j : j + 2]] for j in range(len(phases) - 1)
                ]
                hairs[np.arange(var_start, var_end), pvar_idx] += 1
                hairs[np.arange(var_start, var_end), pvar_idx_rev] += 1
    print(f"total processed {ctr} reads")
    return hairs


def load_hairs(hair_file: str, smoothing=True, alpha=1):
    """
    load (nsnp, 4) hair tsv file, +alpha smoothing
    """
    hairs = None
    with gzip.open(hair_file, "rt") as f:
        hairs = np.loadtxt(f, delimiter="\t", dtype=np.int16)
    hairs = hairs.astype(np.float32)
    if smoothing:
        hairs[:, :] += alpha
    return hairs


# path = None
# files = os.listdir(path)
# samples = set(file.split("_")[0] for file in files)
# for sample in samples:
#     normal = f"{sample}_Normal.hairs.tsv.gz"
#     tumor = f"{sample}_Tumor.hairs.tsv.gz"
#     if os.path.exists(normal) and os.path.exists(tumor):
#         hairsN = load_hairs(normal)
#         hairsT = load_hairs(tumor)
#         hairs_merged = hairsN + hairsT
#         with gzip.open(os.path.join(path, f"{sample}_merged.hairs.tsv.gz"), "wt") as f:
#             np.savetxt(f, hairs_merged, fmt="%d", delimiter="\t")
#             f.close()
