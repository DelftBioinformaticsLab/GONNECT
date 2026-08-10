import pandas as pd
import scanpy as sc
import anndata
from thesis_binn.data_processing.data_preprocessing import read_gene_names_to_uniprot_ids, read_gene_names_and_ids, \
    read_gene_ids, read_uniprot_ids_to_gene_names, save_list


def test_mapping(n):
    print(f"filtered pair: {filtered_ids[n]} - {filtered_genes[n]}")
    print(f"id2name  pair: {filtered_ids[n]} - {id2name[filtered_ids[n]]}")
    print(f"name2id  pair: {filtered_genes[n]} - {name2id[filtered_genes[n]]}")


if __name__ == '__main__':
    save = False
    data_folder = "../../../../GO_TCGA"
    # Load data
    dataset_name = "TCGA_complete"
    data = pd.read_pickle(f"{data_folder}/expression.pkl")
    metadata = pd.read_csv(f"{data_folder}/clinical.csv")

    # Extract gene names from data
    n_nan_cols = 2
    full_gene_list = list(data.columns[n_nan_cols:])

    # Store gene name list
    if save:
        save_list(list(full_gene_list), f"{data_folder}/{dataset_name}_full_gene_list.txt")

    # Get Uniprot IDs for gene names
    name2id = read_gene_names_to_uniprot_ids(path=f"{data_folder}/{dataset_name}_name2uniprotkb.tsv")
    id2name = read_uniprot_ids_to_gene_names(path=f"{data_folder}/{dataset_name}_name2uniprotkb.tsv")
    tcga_names, tcga_ids = read_gene_names_and_ids(path=f"{data_folder}/{dataset_name}_name2uniprotkb.tsv")
    go_ids = read_gene_ids(f"{data_folder}/gene_go_bp_id.txt")

    # Find which Uniprot IDs have GO BP annotations
    overlapping_ids = set(tcga_ids).intersection(set(go_ids))

    # Find to which genes these overlapping IDs belong
    genes_in_go = [tcga_names[i] for i in range(len(tcga_names)) if tcga_ids[i] in overlapping_ids]
    ids_in_go = [tcga_ids[i] for i in range(len(tcga_names)) if tcga_ids[i] in overlapping_ids]

    # Are there any genes that appear more than once?
    # This would mean that one gene has multiple IDs in GO
    print(f"genes_in_go: {len(genes_in_go)}, set(genes_in_go): {len(set(genes_in_go))}")
    # Yes, so we need to make a list of one-to-one mappings from genes to IDs
    # First make dict of IDs sorted by number of duplicates (so multiple genes link ot the same ID)
    id_match_counts = {}
    for gene_id in ids_in_go:
        match_count = ids_in_go.count(gene_id)
        if id_match_counts.keys().__contains__(match_count):
            if gene_id not in id_match_counts[match_count]:
                id_match_counts[match_count].append(gene_id)
        else:
            id_match_counts[match_count] = [gene_id]

    # Match all IDs to only a single gene, in ascending order of #duplicates
    filtered_genes = []
    filtered_ids = []
    for id_match_count in sorted(id_match_counts.keys()):
        duplicate_ids = id_match_counts[id_match_count]
        # Loop over all IDs with i duplicates
        for duplicate_id in duplicate_ids:
            linked_genes = id2name[duplicate_id]
            # Loop over all genes associated to this ID
            for gene in linked_genes:
                # Find a gene that is still available and link ID to only that gene
                # If no gene is available, this ID is dismissed
                if gene not in filtered_genes:
                    filtered_genes.append(gene)
                    filtered_ids.append(duplicate_id)
                    break

    # filtered_genes and filtered_ids should now be one-to-one
    print(f"filtered_ids: {len(filtered_ids)}, set(filtered_ids): {len(set(filtered_ids))}")
    print(f"filtered_genes: {len(filtered_genes)}, set(filtered_genes): {len(set(filtered_genes))}")

    # Filter data for annotated genes
    filtered_cols = ["patient_id", "sample_type"] + filtered_genes
    filtered_data = data[filtered_cols]

    # Store mapping from Uniprot ID to gene name
    if save:
        with open(f"{data_folder}/{dataset_name}_gene_id_pairs_in_go_bp.txt", "w") as f:
            for i in range(len(filtered_genes)):
                f.write(filtered_ids[i] + "\t" + filtered_genes[i] + "\n")

    # Rename columns from gene name to Uniprot ID
    filtered_data.columns = ["patient_id", "sample_type"] + filtered_ids

    # After renaming, genes can be ordered alphabetically
    filtered_ids = sorted(filtered_ids)

    # Add columns with prediction labels from metadata
    labels = metadata[["patient_id", "cancer_type", "tumor_tissue_site", "stage_pathologic_stage"]]
    filtered_data_with_labels = filtered_data.merge(labels, on="patient_id", how="left")
    cols_in_order = ["patient_id", "sample_type", "cancer_type", "tumor_tissue_site",
                     "stage_pathologic_stage"] + filtered_ids
    filtered_data_with_labels = filtered_data_with_labels[cols_in_order]

    # Remove healthy tissue samples
    filtered_data_with_labels_normal_tissue_removed = filtered_data_with_labels[
        ~filtered_data_with_labels['sample_type'].str.contains('normal', case=False, na=False)]

    # Drop genes with zero variance
    filtered_data_varless_genes = filtered_data_with_labels_normal_tissue_removed.copy()
    filtered_ids_with_var = [col for col in filtered_ids if filtered_data_varless_genes[col].std() != 0]
    filtered_data_varless_genes_removed = filtered_data_varless_genes[
        ["patient_id", "sample_type", "cancer_type", "tumor_tissue_site",
         "stage_pathologic_stage"] + filtered_ids_with_var]

    # Select n most variable genes
    n = 1000
    data_clean = filtered_data_varless_genes_removed.copy()
    adata = anndata.AnnData(data_clean[filtered_ids_with_var].to_numpy())
    adata.var_names = filtered_ids_with_var

    sc.pp.highly_variable_genes(adata, n_top_genes=n, flavor="seurat")
    top_n_ids = adata.var[adata.var['highly_variable']].index.tolist()
    data_clean_top_n = data_clean[
        ["patient_id", "sample_type", "cancer_type", "tumor_tissue_site", "stage_pathologic_stage"] + sorted(top_n_ids)]

    # Normalize data
    data_clean_top_n[top_n_ids] = (data_clean_top_n[top_n_ids] - data_clean_top_n[top_n_ids].mean()) / \
                                  data_clean_top_n[top_n_ids].std()
    # Store the final dataset
    if save:
        print("Start saving dataset.")
        final_dataset = data_clean_top_n
        final_dataset.to_csv(f"../../../data/{dataset_name}_bp_top1k.csv.gz", compression="gzip", index=False)
        print("Dataset saved successfully.")
