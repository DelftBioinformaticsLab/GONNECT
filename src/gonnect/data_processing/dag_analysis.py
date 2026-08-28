from goatools.obo_parser import GOTerm
import matplotlib.pyplot as plt
import numpy as np

from gonnect.data_processing.GeneTerm import GeneTerm
from gonnect.data_processing.ProxyTerm import ProxyTerm


def is_alternative_id(go: dict[str, GOTerm], term_id):
    """Returns True if the given ID is a pseudonym of a GO term object"""
    return term_id != go[term_id].item_id


def is_imbalanced(term: GOTerm) -> bool:
    """Returns True if the given GO term is imbalanced"""
    return term.level != term.depth


def all_go_leaf_ids(go: dict[str, GOTerm]):
    """Returns all term IDs with 0 child terms. Genes do not count as children."""
    leaf_ids = set()
    for term_id in go.keys():
        if isinstance(go[term_id], GeneTerm) or isinstance(go[term_id], ProxyTerm):
            continue
        is_leaf = True
        for child in go[term_id].children:
            if not (isinstance(child, GeneTerm) or isinstance(child, ProxyTerm)):
                is_leaf = False
        if is_leaf:
            leaf_ids.add(term_id)
    return leaf_ids


def all_leaf_ids(go: dict[str, GOTerm]):
    """Returns all term IDs with no children."""
    leaf_ids = set()
    for term_id in go.keys():
        if len(go[term_id].children) == 0:
            leaf_ids.add(term_id)
    return leaf_ids


def direct_parents(go: dict[str, GOTerm], term_ids):
    """Returns a set of all the immediate parents from the IDs in the input set."""
    parent_set = set()
    for term_id in term_ids:
        term = go[term_id]
        parents = set([parent.item_id for parent in term.parents])
        parent_set = parent_set.union(parents)
    return parent_set


def layers_with_duplicates(go: dict[str, GOTerm], only_id=False):
    """Returns a list of sets containing leaves and consecutive immediate parents.
    Terms can appear multiple times in different layers. Argument determines whether
    layers consist of objects (GOTerm) or IDs (String)."""
    layers = []
    leaves = all_leaf_ids(go)

    if only_id:
        layers.append(leaves)
    else:
        layers.append({go[term_id] for term_id in leaves})

    parents = direct_parents(go, leaves)
    while len(parents) > 0:
        if only_id:
            layers.append(parents)
        else:
            layers.append({go[term_id] for term_id in parents})
        parents = direct_parents(go, parents)
    return layers


def number_of_children(go: dict[str, GOTerm], term_ids):
    """Returns a dictionary where the terms in the input set
    are sorted by total number of children (recursively)."""
    terms_by_number_of_children = {}
    for term_id in term_ids:
        children = go[term_id].get_all_children()
        if len(children) in terms_by_number_of_children:
            terms_by_number_of_children[len(children)] = terms_by_number_of_children[len(children)].union([go[term_id]])
        else:
            terms_by_number_of_children[len(children)] = {go[term_id]}
    return terms_by_number_of_children


def layer_overlap(layers):
    """Returns a dict containing the intersection of each possible pair of layers"""
    overlap = {}
    for k in range(len(layers) - 1):
        for i in range(k + 1, len(layers)):
            if i == k:
                pass
            else:
                overlap[(k, i)] = layers[k].intersection(layers[i])
    return overlap


def plot_depth_distribution(go: dict[str, GOTerm], term_ids, sub_fig=None, alpha=0.5, bins=np.arange(20) - 0.5,
                            title="Distribution of GO-term depths", show_proxy=False):
    depths = []
    max_depth_term_id = "GO:0000001"
    min_depth_term_id = "GO:0000001"
    for term_id in term_ids:
        if not show_proxy:
            if isinstance(go[term_id], ProxyTerm):
                continue
        depths.append(go[term_id].depth)
        if go[term_id].depth > go[max_depth_term_id].depth:
            max_depth_term_id = term_id
        if go[term_id].depth < go[min_depth_term_id].depth:
            min_depth_term_id = term_id

    if sub_fig:
        sub_fig.set_xlabel("Depth")
        sub_fig.set_ylabel("Number of terms")
        sub_fig.set_xticks(np.arange(start=0, stop=20, step=2))
        sub_fig.set_title(title)
        sub_fig.hist(depths, alpha=alpha, bins=bins, edgecolor="k")
    else:
        plt.xlabel("Depth")
        plt.ylabel("Number of terms")
        plt.xticks(np.arange(start=0, stop=20, step=2))
        plt.title(title)
        plt.hist(depths, alpha=alpha, bins=bins, edgecolor="k")


def print_dag_info(dag: dict[str, GOTerm]):
    terms = len(dag.values())
    proxies = len([term for term in dag.values() if isinstance(term, ProxyTerm)])
    genes = len([term for term in dag.values() if isinstance(term, GeneTerm)])
    print(f"\nNumber of nodes: {terms}")
    print(f"Number of leafs: {len(all_leaf_ids(dag))}")
    print(f"number of genes: {genes}")
    print(f"Proxy terms: {proxies}/{terms} = {proxies / terms * 100:.1f}%")
    print(f"Max depth: {max(term.depth for term in dag.values())}")


def create_layers_deprecated(dag: dict[str, GOTerm], root_id="GO:0000000"):
    """Return a list of sorted collections of nodes with equal depth.
    Deprecated: Only works for balanced DAGs (duplicates otherwise), unnecessarily complex."""
    layers = [[dag[root_id]]]
    next_layer = sorted(list(dag[root_id].children), key=lambda x: x.item_id)
    layers.append(next_layer)
    prev_layer = next_layer.copy()
    while len(next_layer) > 0:
        next_layer = set()
        for node in prev_layer:
            next_layer.update(node.children)
        next_layer = sorted(list(next_layer), key=lambda x: x.item_id)
        layers.append(next_layer)
        prev_layer = next_layer.copy()
    return layers[:-1]


def create_layers(dag: dict[str, GOTerm]):
    """Returns a list of terms per depth level. NB: based on depth, not on edges.
    Therefore, indicative but not representable for imbalanced graphs."""
    layers = dict()
    min_depth = 0
    for term_id in dag.keys():
        # Store minimal depth for sorting layers by depth
        if dag[term_id].depth < min_depth:
            min_depth = dag[term_id].depth
        # Add terms to corresponding depth key
        if dag[term_id].depth in layers.keys():
            layers[dag[term_id].depth].add(dag[term_id])
        else:
            layers[dag[term_id].depth] = set()
            layers[dag[term_id].depth].add(dag[term_id])

    # Convert dictionary to list
    layer_list = []
    for i in range(len(layers.keys())):
        # Sort layers by depth, and terms in layer by ID
        layer = layers[min_depth + i]
        sorted_layer = sorted(list(layer), key=lambda x: x.item_id)
        layer_list.append(sorted_layer)
    return layer_list


def print_layers(layers, show_visualization=True):
    """Prints the lengths of the depth-based DAG layers, for different types of terms.
    Possibility to visualize the structure by printing its shape."""
    go_terms, gene_terms, proxy_terms = [], [], []
    n_go, n_gene, n_proxy = 0, 0, 0
    n = 100
    max_layer_len = max(len(layer) for layer in layers)
    symbol_multiplier = n / max_layer_len
    for i, layer in enumerate(layers):
        go = []
        gene = []
        proxy = []
        for term in layer:
            if isinstance(term, ProxyTerm):
                proxy.append(term)
            elif isinstance(term, GeneTerm):
                gene.append(term)
            else:
                go.append(term)
        go_terms.append(go)
        gene_terms.append(gene)
        proxy_terms.append(proxy)
        n_go += len(go)
        n_gene += len(gene)
        n_proxy += len(proxy)
    if show_visualization:
        print((n + 3) * " " + f"Total: terms/genes/proxies\t{n_go}/{n_gene}/{n_proxy}")
        for i in range(len(layers)):
            empty_symbols = int((max_layer_len - len(layers[i])) * symbol_multiplier)
            go_symbols = int(len(go_terms[i]) * symbol_multiplier)
            gene_symbols = int(len(gene_terms[i]) * symbol_multiplier)
            proxy_symbols = int(len(proxy_terms[i]) * symbol_multiplier)
            if len(go_terms[i]) > 0 and go_symbols == 0:
                go_symbols = 1
            if len(gene_terms[i]) > 0 and gene_symbols == 0:
                gene_symbols = 1
            if len(proxy_terms[i]) > 0 and proxy_symbols == 0:
                proxy_symbols = 1
            visualization = int(empty_symbols / 2) * " " + int(
                proxy_symbols / 2) * "'" + go_symbols * "|" + gene_symbols * "*" + int(proxy_symbols / 2) * "'" + int(
                empty_symbols / 2) * " "
            visualization += " " * (n - len(visualization))
            print(
                f"{visualization} \t{i + 1}. {len(layers[i])}: {len(go_terms[i])}/{len(gene_terms[i])}/{len(proxy_terms[i])}")

        print("\t| = GO terms\n\t* = gene terms\n\t' = proxy terms")
    else:
        print(f"   Total: terms/genes/proxies\t{n_go}/{n_gene}/{n_proxy}")
        for i in range(len(go_terms)):
            print(f"{i + 1}. {len(layers)}: {len(go_terms[i])}/{len(gene_terms[i])}/{len(proxy_terms[i])}")


def only_go_terms(dag: dict[str, GOTerm]):
    """Returns all IDs of genuine terms and ignores proxy terms."""
    go_term_ids = []
    for term_id in dag.keys():
        if isinstance(dag[term_id], ProxyTerm):
            continue
        go_term_ids.append(term_id)
    return go_term_ids


def genes_not_on_leaves_ids(dag: dict[str, GOTerm]):
    non_leaf_gene_ids = set()
    for term_id in dag.keys():
        if isinstance(dag[term_id], GeneTerm):
            for parent in dag[term_id].parents:
                for child in parent.children:
                    if not isinstance(child, GeneTerm):
                        non_leaf_gene_ids.add(term_id)
    return non_leaf_gene_ids
