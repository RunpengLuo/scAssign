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
    min_clen_clustering=10e6,
):
    """
    sort cells by 1) clonal states 2) subclonal states
    """
    if not all(seg2clen[cid] >= min_clen_clustering for cid in cluster_ids):
        cluster_ids = [
            cid for cid in cluster_ids if seg2clen[cid] >= min_clen_clustering
        ]
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


def _build_baf_matrix(
    bcounts: np.ndarray, tcounts: np.ndarray, dec_bc_ids=None, get_df=True
):
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
    min_clen_clustering=10e6,
):
    print("plot heatmaps")
    if os.path.exists(out_baf_file) and os.path.exists(out_snp_file):
        return
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
            dec_baf_gex_df = _build_baf_matrix(
                data.gex_bcounts, data.gex_tcounts, dec_bc_ids
            )
            dec_nsnp_gex_df = _build_snp_matrix(data.gex_nsnps, dec_bc_ids)

        dec_baf_atac_df = None
        dec_nsnp_atac_df = None
        if data.has_atac:
            dec_baf_atac_df = _build_baf_matrix(
                data.atac_bcounts, data.atac_tcounts, dec_bc_ids
            )
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
    dec_colname="Decision",
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
            gex_matrix = _build_baf_matrix(
                data.gex_bcounts, data.gex_tcounts, dec_bc_ids, get_df=False
            )  # segment by cell
        atac_matrix = None
        if data.has_atac:
            atac_matrix = _build_baf_matrix(
                data.atac_bcounts, data.atac_tcounts, dec_bc_ids, get_df=False
            )
        dec2baf_mat[dec_name] = [gex_matrix, atac_matrix]

    segs = data.bins
    pdf_fd_1d = PdfPages(out_file)
    for si, seg in segs.iterrows():
        ch, seg_s, seg_t = seg["#CHR"], seg["START"], seg["END"]
        seg_df = pd.DataFrame(columns=["BAF", "Modality", "Decision"])
        # get all cells at this segment, one row per decision
        ncells_per_dec = {dec_name: [0, 0] for dec_name in dec2baf_mat.keys()}
        for dec_name, [gex_mat, atac_mat] in dec2baf_mat.items():
            if data.has_gex:
                gex_vec = gex_mat[si, :]
                gex_vec = gex_vec[gex_vec != -1]
                ncells_per_dec[dec_name][0] = len(gex_vec)
                if len(gex_vec) > 0:
                    dec_df = pd.DataFrame(
                        {"BAF": gex_vec, "Modality": "scRNAseq", dec_colname: dec_name}
                    )
                    seg_df = pd.concat([seg_df, dec_df])
            if data.has_atac:
                atac_vec = atac_mat[si, :]
                atac_vec = atac_vec[atac_vec != -1]
                ncells_per_dec[dec_name][1] = len(atac_vec)
                if len(atac_vec) > 0:
                    dec_df = pd.DataFrame(
                        {
                            "BAF": atac_vec,
                            "Modality": "scATACseq",
                            dec_colname: dec_name,
                        }
                    )
                    seg_df = pd.concat([seg_df, dec_df])

        # all cells in segment si
        ncells_per_dec["All"] = [0, 0]
        if data.has_gex:
            gex_matrix = _build_baf_matrix(
                data.gex_bcounts,
                data.gex_tcounts,
                assign_df.index.to_numpy(),
                get_df=False,
            )
            gex_vec = gex_matrix[si, :]
            gex_vec = gex_vec[gex_vec != -1]
            seg_df = pd.concat(
                [
                    seg_df,
                    pd.DataFrame(
                        {"BAF": gex_vec, "Modality": "scRNAseq", dec_colname: "All"}
                    ),
                ]
            )
            ncells_per_dec["All"][0] = len(gex_vec)

        if data.has_atac:
            atac_matrix = _build_baf_matrix(
                data.atac_bcounts,
                data.atac_tcounts,
                assign_df.index.to_numpy(),
                get_df=False,
            )
            atac_vec = atac_matrix[si, :]
            atac_vec = atac_vec[atac_vec != -1]
            seg_df = pd.concat(
                [
                    seg_df,
                    pd.DataFrame(
                        {"BAF": atac_vec, "Modality": "scATACseq", dec_colname: "All"}
                    ),
                ]
            )
            ncells_per_dec["All"][1] = len(atac_vec)

        if len(seg_df) == 0:
            print(f"no signals found at {ch}:{seg_s}-{seg_t}")
            continue
        seg_df["BAF"] = seg_df["BAF"].astype(np.float32)
        seg_df["Modality"] = seg_df["Modality"].astype("str")
        seg_df[dec_colname] = seg_df[dec_colname].astype("str")
        fig, ax = plt.subplots(nrows=1, ncols=1)
        sns.violinplot(
            data=seg_df,
            x=dec_colname,
            y="BAF",
            hue="Modality",
            ax=ax,
            hue_order=["scRNAseq", "scATACseq"],
        )

        original_labels = [tick.get_text() for tick in ax.get_xticklabels()]
        new_labels = [
            f"{label}\n{ncells_per_dec[label][0]};{ncells_per_dec[label][1]}"
            for label in original_labels
        ]
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


############################################################
def plot_cross_heatmap(
    assign_df: pd.DataFrame,
    sample: str,
    outfile: str,
    acol="final_type",
    bcol="Decision",
):
    """
    Plot heatmap to cross-check assignments and other method's result
    """
    avals = assign_df[acol].unique().tolist()
    bvals = assign_df[bcol].unique().tolist()
    num_avals = len(avals)
    num_bvals = len(bvals)
    data = pd.pivot_table(
        assign_df, index=acol, columns=bcol, aggfunc="size", fill_value=0
    ).astype(int)
    data = data.astype(int)
    print(data)

    fig, axes = plt.subplots(
        num_avals,
        1,
        figsize=(6, 6),
        gridspec_kw={"height_ratios": [1] * num_avals},
        # constrained_layout=True
    )

    if num_avals == 1:
        axes = [axes]
    for i, aval in enumerate(avals):
        row = np.array(data.loc[aval, bvals].tolist()).reshape(1, num_bvals)
        ax = axes[i]
        im = ax.imshow(
            row, aspect="auto", cmap="RdYlGn", vmin=np.min(row), vmax=np.max(row)
        )
        for j in range(num_bvals):
            ax.text(
                j,
                0,
                row[0, j],
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                fontsize=24,
            )
        if i != num_avals - 1:
            ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(aval, rotation=45, labelpad=50)
        cbar = plt.colorbar(im, ax=ax, orientation="vertical", fraction=0.02, pad=0.02)
        cbar.ax.tick_params(labelsize=8)

    axes[0].set_title(
        f"Cell Assignment Heatmap {sample}", fontweight="bold", fontsize=18
    )
    axes[-1].set_xticks(np.arange(num_bvals), labels=bvals)
    axes[-1].tick_params("x", length=0, width=0, gridOn=False, left=False, right=False)
    plt.tight_layout()

    plt.savefig(outfile, dpi=300)
    plt.close()
    return


############################################################
def plot_1d_scatter(
    sample: str,
    data: SC_Data,
    assign_df: pd.DataFrame,  # (cell, clone)
    sz_file: str,
    out_file: str,
    modality=None,
    dec_colname="Decision",
):
    """
    For each decision
        1) for each bin, aggregate b-counts and t-counts
        2) compute per-bin aggregated BAF
    Plot 1d scatter along the chromosomes
        1) BAF
        2) chromosome boundary
        3) expected BAF
    """
    assert modality in ["ATAC", "GEX"]
    assert dec_colname in assign_df.columns
    if modality == "ATAC":
        bbc = data.atac_bins
        bcounts = data.atac_bcounts
        tcounts = data.atac_tcounts
    else:
        bbc = data.gex_bins
        bcounts = data.gex_bcounts
        tcounts = data.gex_tcounts

    ################
    # prepare bounderies
    sz_df = pd.read_table(sz_file, sep="\t", names=["#CHR", "LENGTH"])
    chrs = sz_df["#CHR"].tolist()[:-2]  # ignore sex-chromosome
    sz_df = sz_df.set_index(keys=["#CHR"])
    chr_left_shift = int(20e6)
    chr_offsets = OrderedDict()
    for i, ch in enumerate(chrs):
        if i == 0:
            # slightly shift chr1 to right
            chr_offsets[ch] = chr_left_shift
        else:
            prev_ch = chrs[i - 1]
            offset = chr_offsets[prev_ch] + sz_df.loc[prev_ch, "LENGTH"]
            chr_offsets[ch] = offset
    chr_end = chr_offsets[chrs[-1]] + sz_df.loc[chrs[-1], "LENGTH"]
    xlab_chrs = chrs  # ignore first dummy variable
    xtick_chrs = []
    for i in range(len(chrs)):
        l = chr_offsets[chrs[i]]
        if i < len(chrs) - 2:
            r = chr_offsets[chrs[i + 1]]
        else:
            r = chr_end
        xtick_chrs.append((l + r) / 2)

    ################
    # compute global position on scatter plot
    bbc.loc[:, "position"] = bbc.apply(
        func=lambda r: chr_offsets[r["#CHR"]] + (r.START + r.END) // 2, axis=1
    )
    bbc.loc[:, "abs-start"] = bbc.apply(
        func=lambda r: chr_offsets[r["#CHR"]] + r.START, axis=1
    )
    bbc.loc[:, "abs-end"] = bbc.apply(
        func=lambda r: chr_offsets[r["#CHR"]] + r.END, axis=1
    )

    segs = data.bins
    segs.loc[:, "position"] = segs.apply(
        func=lambda r: chr_offsets[r["#CHR"]] + (r.START + r.END) // 2, axis=1
    )
    segs.loc[:, "abs-start"] = segs.apply(
        func=lambda r: chr_offsets[r["#CHR"]] + r.START, axis=1
    )
    segs.loc[:, "abs-end"] = segs.apply(
        func=lambda r: chr_offsets[r["#CHR"]] + r.END, axis=1
    )

    ################
    # compute aggregated BAF per decision
    dec_names = sorted(assign_df[dec_colname].unique())
    dec2dec_counts = {}
    for dec_name in dec_names:
        dec_bc_ids = assign_df[assign_df[dec_colname] == dec_name].index.to_numpy()
        dec2dec_counts[dec_name] = len(dec_bc_ids)
        agg_bcounts = np.sum(bcounts[:, dec_bc_ids], axis=1)
        agg_tcounts = np.sum(tcounts[:, dec_bc_ids], axis=1)
        # per-block cell-aggregated BAF
        bafs = np.divide(
            agg_bcounts,
            agg_tcounts,
            out=np.full_like(agg_tcounts, fill_value=np.nan, dtype=np.float32),
            where=agg_tcounts > 0,
        )
        bbc.loc[:, dec_name] = bafs

    # filter any bin/decision nan entry
    bbc = bbc.loc[~bbc.isna().any(axis=1), :]

    ################
    # add expected copy-number for Decision
    if dec_colname == "Decision":
        exp_baf_lines = {}
        bl_colors = {}
        clone2i = {clone: i for i, clone in enumerate(data.clones)}
        for dec_name in dec_names:
            exp_baf_lines[dec_name] = []
            bl_colors[dec_name] = []
            if dec_name == "unassigned":
                continue
            for i, seg in segs.iterrows():
                exp_baf = data.bafs[i, clone2i[dec_name]]
                exp_baf_lines[dec_name].append(
                    [(seg["abs-start"], exp_baf), (seg["abs-end"], exp_baf)]
                )
                bl_colors[dec_name].append((0, 0, 0, 1))

    ################
    # prepare platte and markers
    markersize = float(max(20, 4 - np.floor(len(bbc) / 500)))
    # markersize_centroid = 10
    # marker_bd_width = 0.8
    sns.set_style("whitegrid")
    palette = sns.color_palette("husl")
    if len(dec_names) > 8:
        palette = sns.color_palette("husl", n_colors=len(dec_names))
    else:
        palette = sns.color_palette("Set2", n_colors=len(dec_names))
    sns.set_palette(palette)

    pdf_fd = PdfPages(out_file)
    for dec_idx, dec_name in enumerate(dec_names):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,4))
        ax.scatter(
            bbc["position"],
            bbc[dec_name],
            s=markersize,
            edgecolors="black",
            linewidths=0.5,
            alpha=0.8,
            color=palette[dec_idx],
            marker='o'  # ensure filled circle
        )
        ax.vlines(
            list(chr_offsets.values()),
            ymin=0,
            ymax=1,
            transform=ax.get_xaxis_transform(),
            linewidth=0.5,
            colors="k",
        )
        # add BAF 0.5 line
        ax.hlines(
            y=0.5,
            xmin=0,
            xmax=chr_end,
            colors="grey",
            linestyle=":",
            linewidth=1,
        )
        if dec_colname == "Decision":
            if len(exp_baf_lines[dec_name]) > 0:
                ax.add_collection(
                    LineCollection(exp_baf_lines[dec_name], linewidth=2, colors=bl_colors[dec_name])
                )
        ax.grid(False)
        plt.setp(ax, xlim=(0, chr_end), xticks=xtick_chrs, xlabel="")
        ax.set_xticklabels(xlab_chrs, rotation=60, fontsize=8)
        ax.set_ylabel("BAF")
        num_dec_cells = dec2dec_counts[dec_name]
        ax.set_title(f"B-allele Frequency Plot - {sample}\n{dec_name} #{num_dec_cells}")
        fig.tight_layout()
        pdf_fd.savefig(fig, dpi=150)
        fig.clear()
        plt.close()
    pdf_fd.close()
    return
