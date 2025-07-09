import os
import sys

from collections import OrderedDict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from scipy.cluster.hierarchy import linkage, leaves_list

from sc_data import SC_Data

# def plot_feature_heatmap(data: SC_Data, feature_type: str):

############################################################
def plot_feature_distribution(mat: np.ndarray, feature_type: str, out_dir: str):
    all_data = mat.flatten()

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.hist(all_data, bins=100, log=True, color="purple", alpha=0.7)
    plt.title(f"Histogram of {feature_type} Counts per feature per cell")
    plt.xlabel("Count")
    plt.ylabel("Frequency (log scale)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"histogram.{feature_type}.png"), dpi=300)
    return

############################################################
def _prep_segs(
    segs_df: pd.DataFrame,
):
    """
    get segment ylabels and colors
    mark different color for LOH clonal, clonal, and sublconal segments
    label segment as clonal or subclonal for clustering purpose
    """
    seg2ylabels = []
    seg2ycolors = []
    seg2cns = []
    seg2clen = []
    for _, seg in segs_df.iterrows():
        ch, seg_s, seg_t = seg["#CHR"], seg["START"], seg["END"]
        cnstr: str = seg["cn-state"]
        cns = cnstr.split(";")
        seg_size = round((seg_t - seg_s) / 1e6)
        seg2ylabels.append(f"{ch}:{seg_size}M " + ";".join(cns[1:]))
        seg2cns.append(cns[1:])
        seg2clen.append(seg_t - seg_s)
        if len(set(cns[1:])) == 1:
            if cns[1][2] == "0":
                seg2ycolors.append("red")  # clonal LOH
            else:
                seg2ycolors.append("green")  # clonal non-LOH
        else:
            seg2ycolors.append("blue")  # subclonal
    return seg2ylabels, seg2ycolors, seg2cns, seg2clen


def _heatmap(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    ylabels: list,
    ycolors: list,
    mask=None,
    cmap=None,
    norm=None,
    stack_plot=True,
    vrange=None,
    xlabs=None,
    title=None,
    is_robust=False,
):
    """
    aux function to plot heatmap
    """
    height_per_row = 0.3
    height_per_half = 0.15
    fig_height = max(4, 1 * len(ylabels) * height_per_row)
    if not stack_plot:
        fig, axes = plt.subplots(1, 2, figsize=(10, fig_height))
        for i, [df, add_cbar] in enumerate([[df1, False], [df2, True]]):
            if df is None:
                continue
            sns.heatmap(
                df,
                ax=axes[i],
                yticklabels=True,
                xticklabels=False,
                mask=(df == mask),
                vmin=vrange[0],
                vmax=vrange[1],
                cbar=add_cbar,
                cmap=cmap,
                robust=is_robust,
            )
            axes[i].set_xlabel(xlabs[i])
        axes[0].set_yticklabels(ylabels, rotation=90)
        for ticklabel, color in zip(axes[0].get_yticklabels(), ycolors):
            ticklabel.set_color(color)
        axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0)
        axes[1].set_yticks([])
        axes[1].set_yticklabels("")
        fig.suptitle(title)
        fig.tight_layout()
        return fig
    else:
        assert len(df1) == len(df2)
        df1 = df1.reset_index(drop=True)
        df2 = df2.reset_index(drop=True)
        # stable sorting to ensure interleaved rows
        interleaved = (
            pd.concat([df1, df2]).sort_index(kind="mergesort").reset_index(drop=True)
        )
        fig, ax = plt.subplots(1, 1, figsize=(10, fig_height))
        sns.heatmap(
            interleaved,
            ax=ax,
            yticklabels=True,
            xticklabels=False,
            mask=(interleaved == mask),
            vmin=vrange[0],
            vmax=vrange[1],
            cbar=True,
            cmap=cmap,
            robust=is_robust,
        )
        ax.set_xlabel(f"{xlabs[0]}/{xlabs[1]}")
        yticks = [i * 2 + 1 for i in range(len(ylabels))]
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, rotation=0, va="center")
        for ticklabel, color in zip(ax.get_yticklabels(), ycolors):
            ticklabel.set_color(color)
        fig.suptitle(title)
        fig.tight_layout()
        return fig

def _cluster_cols(
    gex_df: pd.DataFrame,
    atac_df: pd.DataFrame,
    seg2clen: list,
    cluster_ids: list,
    min_clen_clustering=10e6
):
    """
    sort cells by 1) clonal states 2) subclonal states
    """
    if not all(seg2clen[cid] >= min_clen_clustering for cid in cluster_ids):
        cluster_ids = [cid for cid in cluster_ids if seg2clen[cid] >= min_clen_clustering]
    if gex_df is None:
        combined_df = atac_df.iloc[cluster_ids, :]
    elif atac_df is None:
        combined_df = gex_df.iloc[cluster_ids, :]
    else:
        combined_df = pd.concat(
            [
                gex_df.iloc[cluster_ids, :],
                atac_df.iloc[cluster_ids, :],
            ],
            axis=0,
        )

    if len(combined_df.T) > 5:
        print(f"\tclustering with {len(combined_df)} segments")
        linkage_cols = linkage(combined_df.T, method="ward")
        col_order = leaves_list(linkage_cols)
        return True, col_order
    else:
        print("\tskip clustering")
        return False, None

def _build_baf_matrix(bcounts: np.ndarray, tcounts: np.ndarray, dec_bc_ids=None, get_df=True):
    dec_bmat = bcounts[:, dec_bc_ids]
    dec_tmat = tcounts[:, dec_bc_ids]
    dec_baf = np.divide(
        dec_bmat,
        dec_tmat,
        out=np.full_like(dec_bmat, fill_value=-1, dtype=np.float32),
        where=dec_tmat != 0,
    )
    if get_df:
        dec_baf_df = pd.DataFrame(dec_baf, columns=range(len(dec_bc_ids)))
        return dec_baf_df
    else:
        return dec_baf

def _build_snp_matrix(nsnp_counts: np.ndarray, dec_bc_ids=None):
    dec_nsnp = nsnp_counts[:, dec_bc_ids]
    dec_nsnp_df = pd.DataFrame(dec_nsnp, columns=range(len(dec_bc_ids)))
    return dec_nsnp_df

def plot_heatmap(
    sample: str,
    data: SC_Data,
    assign_df: pd.DataFrame,  # (cell, clone)
    out_baf_file: str,
    out_snp_file: str,
    dec_colname="Decision",
    ignore_clonal=False,
    min_clen_clustering=10e6
):
    print("plot heatmaps")
    # if os.path.exists(out_baf_file) and os.path.exists(out_snp_file):
    #     return
    seg2ylabels, seg2ycolors, seg2cns, seg2clen = _prep_segs(data.bins)
    is_clonal = [len(set(cns)) == 1 for cns in seg2cns]
    clonal_ids = [i for i in range(len(seg2cns)) if is_clonal[i]]
    subclonal_ids = [i for i in range(len(seg2cns)) if not is_clonal[i]]
    cluster_ids = subclonal_ids if len(subclonal_ids) != 0 else clonal_ids
    if len(subclonal_ids) > 0 and ignore_clonal:
        seg2ylabels = [seg2ylabels[i] for i in subclonal_ids]
        seg2ycolors = [seg2ycolors[i] for i in subclonal_ids]

    bounds = np.linspace(0, 1, 11)  # 11 edges = 10 bins
    n_bins = len(bounds) - 1
    base_cmap = plt.get_cmap("viridis")
    colors = base_cmap(np.linspace(0, 1, 10))
    cmap = LinearSegmentedColormap.from_list("viridis_10", colors, N=n_bins)
    norm = BoundaryNorm(bounds, ncolors=cmap.N)

    cmap.set_bad(color="gray")

    pdf_fd_baf = PdfPages(out_baf_file)
    pdf_fd_snp = PdfPages(out_snp_file)
    for dec_name in sorted(assign_df[dec_colname].unique()):
        print(f"plot {dec_name}")
        dec_bc_ids = assign_df[assign_df[dec_colname] == dec_name].index.to_numpy()
        num_bcs = len(dec_bc_ids)
        dec_baf_gex_df = None
        dec_nsnp_gex_df = None
        if data.has_gex:
            dec_baf_gex_df = _build_baf_matrix(data.gex_bcounts,
                                               data.gex_tcounts,
                                               dec_bc_ids)
            dec_nsnp_gex_df = _build_snp_matrix(data.gex_nsnps, dec_bc_ids)

        dec_baf_atac_df = None
        dec_nsnp_atac_df = None
        if data.has_atac:
            dec_baf_atac_df = _build_baf_matrix(data.atac_bcounts,
                                               data.atac_tcounts,
                                               dec_bc_ids)
            dec_nsnp_atac_df = _build_snp_matrix(data.atac_nsnps, dec_bc_ids)

        clustered, col_order = _cluster_cols(
            dec_baf_gex_df, dec_baf_atac_df, seg2clen, cluster_ids, min_clen_clustering
        )
        if clustered:
            if data.has_gex:
                dec_baf_gex_df = dec_baf_gex_df.iloc[:, col_order]
                dec_nsnp_gex_df = dec_nsnp_gex_df.iloc[:, col_order]
            if data.has_atac:
                dec_baf_atac_df = dec_baf_atac_df.iloc[:, col_order]
                dec_nsnp_atac_df = dec_nsnp_atac_df.iloc[:, col_order]

        if len(subclonal_ids) > 0 and ignore_clonal:
            if data.has_gex:
                dec_baf_gex_df = dec_baf_gex_df.iloc[subclonal_ids, :]
                dec_nsnp_gex_df = dec_nsnp_gex_df.iloc[subclonal_ids, :]
            if data.has_atac:
                dec_baf_atac_df = dec_baf_atac_df.iloc[subclonal_ids, :]
                dec_nsnp_atac_df = dec_nsnp_atac_df.iloc[subclonal_ids, :]

        fig = _heatmap(
            dec_baf_gex_df,
            dec_baf_atac_df,
            seg2ylabels,
            seg2ycolors,
            -1,
            cmap,
            norm,
            False,
            [0.0, 1.0],
            ["GEX", "ATAC"],
            title=f"{sample} {dec_name} segment-level BAF matrix, #cells={num_bcs}",
            is_robust=False,
        )
        pdf_fd_baf.savefig(fig, dpi=150)
        fig.clear()
        plt.close()

        fig = _heatmap(
            dec_nsnp_gex_df,
            dec_nsnp_atac_df,
            seg2ylabels,
            seg2ycolors,
            0,
            None,
            None,
            False,
            [1, None],
            ["GEX", "ATAC"],
            title=f"{sample} {dec_name} segment-level #SNP matrix, #cells={num_bcs}",
            is_robust=True,
        )
        pdf_fd_snp.savefig(fig, dpi=150)
        fig.clear()
        plt.close()
    pdf_fd_baf.close()
    pdf_fd_snp.close()
    return

############################################################
def plot_violin(
    sample: str,
    data: SC_Data,
    assign_df: pd.DataFrame,  # (cell, clone)
    out_file: str,
    dec_colname="Decision"
):
    """
    plot 1 segment per page
    """

    seg2ylabels, seg2ycolors, seg2cns, seg2clen = _prep_segs(data.bins)
    dec2baf_mat = {}
    for dec_name in sorted(assign_df[dec_colname].unique()):
        # if dec_name == "unassigned":
        #     continue
        dec_bc_ids = assign_df[assign_df[dec_colname] == dec_name].index.to_numpy()
        gex_matrix = None
        if data.has_gex:
            gex_matrix = _build_baf_matrix(data.gex_bcounts, data.gex_tcounts, dec_bc_ids, get_df=False) # segment by cell
        atac_matrix = None
        if data.has_atac:
            atac_matrix = _build_baf_matrix(data.atac_bcounts, data.atac_tcounts, dec_bc_ids, get_df=False)
        dec2baf_mat[dec_name] = [gex_matrix, atac_matrix]

    segs = data.bins
    pdf_fd_1d = PdfPages(out_file)
    for si, seg in segs.iterrows():
        ch, seg_s, seg_t = seg["#CHR"], seg["START"], seg["END"]
        seg_df = pd.DataFrame(columns=["BAF", "Modality", "Decision"])
        # get all cells at this segment, one row per decision
        ncells_per_dec = {dec_name: [0,0] for dec_name in dec2baf_mat.keys()}
        for dec_name, [gex_mat, atac_mat] in dec2baf_mat.items():
            if data.has_gex:
                gex_vec = gex_mat[si, :]
                gex_vec = gex_vec[gex_vec != -1]
                ncells_per_dec[dec_name][0] = len(gex_vec)
                if len(gex_vec) > 0:
                    dec_df = pd.DataFrame({"BAF": gex_vec, "Modality": "scRNAseq", dec_colname: dec_name})
                    seg_df = pd.concat([seg_df, dec_df])
            if data.has_atac:
                atac_vec = atac_mat[si, :]
                atac_vec = atac_vec[atac_vec != -1]
                ncells_per_dec[dec_name][1] = len(atac_vec)
                if len(atac_vec) > 0:
                    dec_df = pd.DataFrame({"BAF": atac_vec, "Modality": "scATACseq", dec_colname: dec_name})
                    seg_df = pd.concat([seg_df, dec_df])
    
        # all cells in segment si
        ncells_per_dec["All"] = [0, 0]
        if data.has_gex:
            gex_matrix = _build_baf_matrix(data.gex_bcounts, data.gex_tcounts, assign_df.index.to_numpy(), get_df=False)
            gex_vec = gex_matrix[si, :]
            gex_vec = gex_vec[gex_vec != -1]
            seg_df = pd.concat([seg_df, pd.DataFrame({"BAF": gex_vec, "Modality": "scRNAseq", dec_colname: "All"})])
            ncells_per_dec["All"][0] = len(gex_vec)
        
        if data.has_atac:
            atac_matrix = _build_baf_matrix(data.atac_bcounts, data.atac_tcounts, assign_df.index.to_numpy(), get_df=False)
            atac_vec = atac_matrix[si, :]
            atac_vec = atac_vec[atac_vec != -1]
            seg_df = pd.concat([seg_df, pd.DataFrame({"BAF": atac_vec, "Modality": "scATACseq", dec_colname: "All"})])
            ncells_per_dec["All"][1] = len(atac_vec)

        if len(seg_df) == 0:
            print(f"no signals found at {ch}:{seg_s}-{seg_t}")
            continue
        seg_df["BAF"] = seg_df["BAF"].astype(np.float32)
        seg_df["Modality"] = seg_df["Modality"].astype("str")
        seg_df[dec_colname] = seg_df[dec_colname].astype("str")
        fig, ax = plt.subplots(nrows=1, ncols=1)
        sns.violinplot(data=seg_df, x=dec_colname, y="BAF", hue="Modality", ax=ax, hue_order=["scRNAseq", "scATACseq"])

        original_labels = [tick.get_text() for tick in ax.get_xticklabels()]
        new_labels = [f"{label}\n{ncells_per_dec[label][0]};{ncells_per_dec[label][1]}" for label in original_labels]
        ax.set_xticklabels(new_labels)

        cn_label = seg2ylabels[si]
        plt.suptitle(f"{sample} {cn_label} Violin Plot\n{seg_s}-{seg_t}")
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), title=None)
        fig.tight_layout()

        pdf_fd_1d.savefig(fig, dpi=300)
        fig.clear()
        plt.close()
    pdf_fd_1d.close()
    return
