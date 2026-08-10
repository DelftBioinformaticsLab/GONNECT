import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from thesis_binn.data_processing.GeneTerm import GeneTerm
from thesis_binn.data_processing.go_preprocessing import link_genes_to_go_by_namespace


def list_ids_in_go_per_namespace(go: dict, namespace: str):
    link_genes_to_go_by_namespace(go, "../../data/goa_human.gaf", namespace)
    linked_ids = []
    for term in go.values():
        if isinstance(term, GeneTerm):
            linked_ids.append(term.item_id)
    return linked_ids


def read_gene_names_to_uniprot_ids(path: str):
    with open(path, "r") as f:
        data_pairs = [pair.split() for pair in f.read().splitlines()]
        data_dict = dict()
        for name, uniprot_id in data_pairs:
            if name in data_dict.keys():
                data_dict[name].append(uniprot_id)
            else:
                data_dict[name] = [uniprot_id]
    return data_dict


def read_uniprot_ids_to_gene_names(path: str):
    with open(path, "r") as f:
        data_pairs = [pair.split() for pair in f.read().splitlines()]
        data_dict = dict()
        for name, uniprot_id in data_pairs:
            if uniprot_id in data_dict.keys():
                data_dict[uniprot_id].append(name)
            else:
                data_dict[uniprot_id] = [name]
    return data_dict


def read_gene_names_and_ids(path: str):
    with open(path, "r") as f:
        data_names_ids = f.read().split()
        data_names = [data_names_ids[i] for i in range(0, len(data_names_ids), 2)]
        data_ids = [data_names_ids[i] for i in range(1, len(data_names_ids), 2)]
        return data_names, data_ids


def read_gene_ids(path: str):
    with open(path, "r") as f:
        gene_ids_from_file = f.read().split()
    return gene_ids_from_file


def save_gene_matches(gene_set1: set[str], gene_set2: set[str], path: str):
    matches = sorted(gene_set1.intersection(gene_set2))
    with open(path, "w") as f:
        f.write("\n".join(list(matches)))


def save_list(a, path: str):
    with open(path, "w") as f:
        f.write("\n".join(a))


def split_data_deprecated(data, n_nan_cols, split=0.7, seed=1):
    """Split the given dataframe in train, validation and test sets. The split argument sets the training fraction, the remainder is split 50/50 between validation and test."""
    validation_test_split = 0.5
    # Strip any non-sample column before making the splits
    cols = list(data.columns[n_nan_cols:])
    train_cols, remaining_cols = train_test_split(cols, train_size=split, random_state=seed)
    validation_cols, test_cols = train_test_split(remaining_cols, train_size=validation_test_split, random_state=seed)
    train_set = data[train_cols]
    validation_set = data[validation_cols]
    test_set = data[test_cols]
    return train_set, validation_set, test_set


if __name__ == '__main__':
    save = False
    data_folder = "../../../../GO_TCGA"
    dataset_name = "GE_top1k"
    id_col = 1
    data = pd.read_csv(f"{data_folder}/{dataset_name}.csv")
    # Retrieve all genes in data as gene name and UniProt ID
    name_2_id = read_gene_names_to_uniprot_ids(path=f"{data_folder}/{dataset_name}_name2uniprotkb.tsv")
    names, ids = read_gene_names_and_ids(path=f"{data_folder}/{dataset_name}_name2uniprotkb.tsv")

    # Save list of ID matches between data and GO (no duplicates)
    # Result: different IDs map to the same name
    go_ids = read_gene_ids(f"{data_folder}/gene_go_bp_id.txt")
    if save:
        save_gene_matches(set(go_ids), set(ids), path=f"{data_folder}/{dataset_name}_matches_bp_id.txt")

    # Save list of gene names corresponding to ID matches
    # Result: duplicate names and duplicate IDs
    match_ids = read_gene_ids(f"{data_folder}/{dataset_name}_matches_bp_id.txt")
    match_names = [names[i] for i in range(len(ids)) if ids[i] in match_ids]
    test_distinct_names_with_match = set(match_names)
    match_names_id = [ids[i] for i in range(len(ids)) if ids[i] in match_ids]
    duplicate_match_names = [i for i in set(match_names) if match_names.count(i) > 1]
    duplicate_match_names_id = [i for i in set(match_names_id) if match_names_id.count(i) > 1]
    if save:
        save_list(match_names, path=f"{data_folder}/{dataset_name}_matches_bp_name.txt")

    # Save (name, ID) pairs for matched IDs
    # Result: duplicate names and duplicate IDs
    match_pairs = [names[i] + "\t" + ids[i] for i in range(len(ids)) if ids[i] in match_ids]
    if save:
        save_list(match_pairs, f"{data_folder}/{dataset_name}_matches_bp_name2uniprotkb.txt")

    # Greedy approach of matching names to IDs
    # if a name has an ID not yet used -> match name and ID
    # else -> exclude gene
    # Result: 15143 unique names with unique IDs
    data_ids_greedy = []
    data_names_greedy = list(name_2_id.keys())
    skipped = []
    id_2_name = read_uniprot_ids_to_gene_names(path=f"{data_folder}/{dataset_name}_name2uniprotkb.tsv")
    for data_name in data_names_greedy:
        id_in_go = False
        skipped_because_id_is_taken = False
        for id_option in name_2_id[data_name]:
            if id_option in go_ids:
                if id_option not in data_ids_greedy:
                    data_ids_greedy.append(id_option)
                    id_in_go = True
                    break
                else:
                    skipped_because_id_is_taken = True
        if not id_in_go:
            data_ids_greedy.append(None)
        if not id_in_go and skipped_because_id_is_taken:
            skipped.append(data_name)

    match_pairs = [data_names_greedy[i] + "\t" + data_ids_greedy[i] for i in range(len(data_names_greedy)) if
                   data_ids_greedy[i]]
    if save:
        save_list(match_pairs, f"{data_folder}/{dataset_name}_matches_bp_name2uniprotkb_greedy.txt")

    # Save dataset containing only gene names with greedy ID (actual IDs not in dataset)
    if save:
        names, ids = read_gene_names_and_ids(path=f"{data_folder}/{dataset_name}_matches_bp_name2uniprotkb_greedy.txt")
        filtered_data = data[data["gene symbol"].isin(names)]
        filtered_data.insert(id_col, "gene id", np.array(ids))
        filtered_data.to_csv(f"{data_folder}/{dataset_name}_bp.csv.gz", compression="gzip", index=False)

        # Save dataset containing the first 100 genes, including ID
        first_hundred_rows = filtered_data.head(100)
        first_hundred_rows.to_csv(f"{data_folder}/{dataset_name}_bp_100.csv.gz", compression="gzip", index=False)

    # Test names, ids both for duplicates
    print(len(names))
    print(len(set(names)))
    print(len(ids))
    print(len(set(ids)))
