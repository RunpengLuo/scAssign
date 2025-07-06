## assign single cells to allele-specific copy-number profile

### HATCHet2
1. run short reads HATCHet2
2. run hapCUT2 on sr snps and long reads


### Raw Input
#### single-cell
1. feature matrix `filtered_feature_bc_matrix`
2. allele count matrix from cellsnp-lite
3. SNPs `snps.vcf.gz`
#### bulk
1. copy-number segment profile `seg.ucn.tsv`
2. blocks BAFs `seg.bbc.tsv`
3. WGS SNP counts tumor1.bed
#### aux
1. SNP HAIR supports (bulk BAM + `snps.vcf.gz`)

### Preprocessing
1. process segment by clone copy-number matrix
2. exclude segments with clonal (1,1) or WGD clonal (2,2), or failed criteria
    1. exclude any SNPs and genes belong to this region
3. compute per-segment SNP phasing information via bulk data (cn information is un-related)
4. compute block by clone BAF matrix
5. scRNAseq and scATACseq
    1. aggregate SNP counts, block by cell total/b-allele matrix
    2. gene by cell matrix

### Input
1. block by cell b-allele/total allele count matrix
2. gene by cell gene expression count matrix
3. peak by cell fragment count matrix
4. segment by clone copy-number matrix A, B.
5. segment by clone BAF matrix A, B.
6. cell ids
7. gene ids
8. peak ids
9. segment id + range

