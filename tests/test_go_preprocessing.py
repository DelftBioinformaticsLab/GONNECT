import os
from pathlib import Path
from unittest import SkipTest, TestCase, skipUnless
from goatools.obo_parser import GODag
from gonnect.data_processing.DAGGenerator import DAGGenerator
import matplotlib.pyplot as plt
from gonnect.data_processing.GeneTerm import GeneTerm
from gonnect.data_processing.dag_analysis import is_imbalanced, print_dag_info, create_layers, \
    genes_not_on_leaves_ids, print_layers, is_alternative_id, create_layers_deprecated
from gonnect.data_processing.data_preprocessing import read_gene_ids
from gonnect.data_processing.go_preprocessing import insert_proxy_terms, update_level_and_depth, pull_leaves_down, \
    relationships_to_parents, merge_prune_until_convergence, balance_until_convergence, link_genes_to_go_by_namespace, \
    remove_geneless_branches, create_dag, copy_dag, filter_by_namespace, prune_skip_connections, merge_chains
from gonnect.data_processing.dag_analysis import layers_with_duplicates, layer_overlap, all_leaf_ids, \
    plot_depth_distribution
from gonnect.data_processing.ProxyTerm import ProxyTerm
from gonnect.paths import DATA_DIR_ENV_VAR, GAF_FILENAME, OBO_FILENAME

# Inputs are resolved rather than hardcoded so the suite runs from any working
# directory, and so $GONNECT_DATA_DIR can point at copies outside the repository.
# Falls back to the repository's own data/ directory.
#
# go-basic.obo is committed, so it is normally present. goa_human.gaf is not (it
# is gitignored -- see the README), and neither is GE_matches_bp_id.txt, so the
# tests needing those skip rather than fail. Nothing here is loaded at import
# time: the sdist ships tests/ without data/, and a bare `pytest` on an unpacked
# sdist must skip cleanly instead of erroring during collection.
DATA_DIR = Path(os.environ.get(DATA_DIR_ENV_VAR) or Path(__file__).resolve().parent.parent / "data")
OBO_PATH = DATA_DIR / OBO_FILENAME
GAF_PATH = DATA_DIR / GAF_FILENAME
GENE_IDS_PATH = DATA_DIR / "GE_matches_bp_id.txt"


def _missing(path):
    return f"{path.name} not found in {DATA_DIR}; set ${DATA_DIR_ENV_VAR} to a directory containing it"


requires_gaf = skipUnless(GAF_PATH.is_file(), _missing(GAF_PATH))
requires_gene_ids = skipUnless(GENE_IDS_PATH.is_file(), _missing(GENE_IDS_PATH))

# Populated by setUpModule. Parsing go-basic.obo twice costs a few seconds, so it
# happens once per run and only when tests are actually executed.
go_main = None
go_bp_main = None
go_rel_main = None
go_rel_main_parentless = None
plot = False


def setUpModule():
    global go_main, go_bp_main, go_rel_main, go_rel_main_parentless
    if not OBO_PATH.is_file():
        raise SkipTest(_missing(OBO_PATH))
    go_main = create_dag(str(OBO_PATH))
    go_bp_main = filter_by_namespace(go_main, {"biological_process"})
    go_rel_main = create_dag(str(OBO_PATH), rel=True)
    go_rel_main_parentless = copy_dag(go_rel_main)
    relationships_to_parents(go_rel_main)


class Test(TestCase):

    def test_copy_dag_small(self):
        dag = DAGGenerator.dag1()
        dag_copy = copy_dag(dag)
        dag.pop("D")
        dag["B"].children.add(dag["A"])
        self.assertNotEqual(len(dag_copy["B"].children), len(dag["B"].children))

    def test_copy_dag(self):
        go_copy = copy_dag(go_bp_main)
        original_kv_pairs = [[key, go_bp_main[key]] for key in go_bp_main.keys()]
        copied_kv_pairs = [[key, go_copy[key]] for key in go_copy.keys()]

        for i in range(len(original_kv_pairs)):
            self.assertEqual(original_kv_pairs[i][0], copied_kv_pairs[i][0])
            self.assertEqual(original_kv_pairs[i][1].item_id, copied_kv_pairs[i][1].item_id)
            self.assertEqual(original_kv_pairs[i][1].level, copied_kv_pairs[i][1].level)
            self.assertEqual(original_kv_pairs[i][1].depth, copied_kv_pairs[i][1].depth)
            self.assertEqual(len(original_kv_pairs[i][1].parents), len(copied_kv_pairs[i][1].parents))

    #       A
    #     / |
    #   /   B
    #  | /    \
    #  C       D
    def test_prune_skip_connections_small(self):
        dag = DAGGenerator.dag1()
        greedy_layers = layers_with_duplicates(dag)
        overlap = layer_overlap(greedy_layers)
        overlapping_terms = sum([len(overlap[pair]) for pair in overlap])
        self.assertEqual(overlapping_terms, 1)
        prune_skip_connections(dag)
        greedy_layers_after_pruning = layers_with_duplicates(dag)
        overlap_after_pruning = layer_overlap(greedy_layers_after_pruning)
        overlapping_terms_after_pruning = sum([len(overlap_after_pruning[pair]) for pair in overlap_after_pruning])
        self.assertEqual(overlapping_terms_after_pruning, 0)
        self.assertNotEqual(overlapping_terms, overlapping_terms_after_pruning)

    def test_prune_skip_connections_go(self):
        go = copy_dag(go_bp_main)
        intersections_a = layer_overlap(layers_with_duplicates(go))
        pruning_events = prune_skip_connections(go)
        intersections_b = layer_overlap(layers_with_duplicates(go))
        self.assertEqual(pruning_events, 0)
        self.assertEqual(len(intersections_a), len(intersections_b))

    #      A
    #    / |
    #  /   B
    # | /
    # C
    def test_merge_chains_small(self):
        dag = DAGGenerator.dag2()
        term_before_merge = len(dag.keys())
        merge_events = merge_chains(dag)
        terms_after_merge = len(dag.keys())
        self.assertEqual(merge_events, 1)
        self.assertLess(terms_after_merge, term_before_merge)

    def test_merge_chains_go(self):
        dag = copy_dag(go_bp_main)
        terms_before_merge = len(dag.keys())
        merge_events = merge_chains(dag)
        terms_after_merge = len(dag.keys())
        self.assertGreater(merge_events, 0)
        self.assertLess(terms_after_merge, terms_before_merge)

    def test_single_merge_effect_on_overlap(self):
        dag = copy_dag(go_bp_main)
        layers = layers_with_duplicates(dag)
        overlap = layer_overlap(layers)
        merge_chains(dag, threshold_parents=1, threshold_children=1)
        merged_layers = layers_with_duplicates(dag)
        merged_overlap = layer_overlap(merged_layers)

        overlap_difference = {}
        for key in overlap.keys():
            overlap_difference[key] = len(merged_overlap[key]) - len(overlap[key])

    def test_merge_chains_until_convergence(self):
        dag = copy_dag(go_bp_main)
        # Plot leaf depth distribution before and after merge-prune convergence
        leaf_ids = all_leaf_ids(dag)
        if plot:
            plot_depth_distribution(dag, leaf_ids)
        # First merge, then prune:
        pruning_events = 1
        merge_events = 1
        parent_threshold = 1
        children_threshold = 1
        while pruning_events + merge_events > 0:
            merge_events = merge_chains(dag, parent_threshold, children_threshold)
            pruning_events = prune_skip_connections(dag)
        leaf_ids_post = all_leaf_ids(dag)
        if plot:
            plot_depth_distribution(dag, leaf_ids_post)
            plt.title(f"Distribution of leaf depth after merge-pruning \n"
                      f"(#parents = {parent_threshold}, #children = {children_threshold})")
            plt.show()

    #       A
    #     / |
    #   /   B
    #  | /    \
    #  C       D
    def test_insert_proxy_terms_small(self):
        dag = DAGGenerator.dag1()
        self.assertNotEqual(dag["C"].level, dag["C"].depth)
        insert_proxy_terms(dag, dag["A"], len(dag))
        self.assertEqual(dag["C"].level, dag["C"].depth)

    def test_update_level_and_depth_small(self):
        dag = DAGGenerator.dag1()
        dag["E"] = ProxyTerm("E", {dag["C"]}, set())
        update_level_and_depth(dag["C"])
        self.assertEqual(dag["E"].level, dag["C"].level + 1)
        self.assertEqual(dag["E"].depth, dag["C"].depth + 1)

    def test_insert_proxy_terms_go(self):
        dag = copy_dag(go_bp_main)
        original_size = len(dag)
        go_root = "GO:0000000"
        # Find all imbalanced terms
        imbalanced = sum(is_imbalanced(term) for term in dag.values())
        dag_size = original_size
        print(f"Imbalanced terms: {imbalanced}")
        # Plot baseline
        leaf_ids = all_leaf_ids(dag)
        if plot:
            plot_depth_distribution(dag, leaf_ids)
        # Multiple passes through the graph to account for shifts in balanced branches
        while imbalanced > 0:
            insert_proxy_terms(dag, dag[go_root], original_size)
            imbalanced = sum(is_imbalanced(term) for term in dag.values())
            print(f"Proxy terms added: {len(dag) - dag_size}")
            print(f"Imbalanced terms left: {imbalanced}")
            dag_size = len(dag)

        print(f"Total amount of inserted balancing proxies: {len(dag) - original_size}")

    def test_plot_merge_results(self):
        go_ref = copy_dag(go_bp_main)
        leaf_ids = all_leaf_ids(go_ref)

        # Depth distribution for different merge conditions
        if plot:
            plot_depth_distribution(go_ref, leaf_ids,
                                    title="Distribution of GO-leaf depths\nfor varying merge conditions")
            plt.legend(["Original"], title="(#parents, #children)")
            plt.show()

        merge_conditions = [(0, 0), (1, 1), (1, 2), (1, 3), (2, 3), (2, 4)]
        for merge_condition in merge_conditions:
            go = copy_dag(go_bp_main)
            pruning_events, merge_events = 1, 1
            while pruning_events + merge_events > 0:
                merge_events = merge_chains(go, merge_condition[0], merge_condition[1])
                pruning_events = prune_skip_connections(go)
            if plot:
                plot_depth_distribution(go_ref, leaf_ids)
                plot_depth_distribution(go, leaf_ids,
                                        title="Distribution of GO-leaf depths\nfor varying merge conditions")
                plt.legend(["Original", merge_condition], title="(#parents, #children)")
                plt.show()

    def test_plot_merge_proxy_results(self):
        go_ref = copy_dag(go_bp_main)
        leaf_ids = all_leaf_ids(go_ref)
        # Depth distribution for different merge conditions, with balancing
        if plot:
            plot_depth_distribution(go_ref, leaf_ids,
                                    title="Distribution of GO-leaf depths\nfor varying merge conditions (balanced)")
            plt.legend(["Original"], title="(#parents, #children)")
            plt.show()

        merge_conditions = [(0, 0), (1, 1), (1, 2), (1, 3), (2, 3), (2, 4)]
        for merge_condition in merge_conditions:
            go = copy_dag(go_bp_main)
            pruning_events, merge_events = 1, 1
            while pruning_events + merge_events > 0:
                merge_events = merge_chains(go, merge_condition[0], merge_condition[1])
                pruning_events = prune_skip_connections(go)

            original_size = len(go)
            go_root = "GO:0000000"
            imbalanced = sum(is_imbalanced(term) for term in go.values())
            dag_size = original_size
            while imbalanced > 0:
                insert_proxy_terms(go, go[go_root], original_size)
                imbalanced = sum(is_imbalanced(term) for term in go.values())

                # when imbalanced, yet no more proxies are added, stop (to stop infinite loop)
                if len(go) - dag_size == 0:
                    print("ERROR: Imbalance cannot be resolved, infinite loop terminated!")
                    break

                print(f"Proxy terms added: {len(go) - dag_size}")
                print(f"Imbalanced terms left: {imbalanced}")
                dag_size = len(go)

            if plot:
                plot_depth_distribution(go_ref, leaf_ids)
                plot_depth_distribution(go, leaf_ids,
                                        title="Distribution of GO-leaf depths\nfor varying merge conditions (balanced)")
                plt.legend(["Original", merge_condition], title="(#parents, #children)")
                plt.show()

    def test_pull_leaves_down_small(self):
        dag = DAGGenerator.dag3()
        pull_leaves_down(dag, len(dag))
        self.assertEqual(dag["C"].depth, dag["E"].depth)

    def test_pull_leaves_down_go(self):
        go = copy_dag(go_bp_main)
        pull_leaves_down(go, len(go))
        print_dag_info(go)
        max_depth = max(go[leaf_id].depth for leaf_id in all_leaf_ids(go))
        depth_sum = sum(go[leaf_id].depth for leaf_id in all_leaf_ids(go))
        self.assertEqual(depth_sum, max_depth * len(all_leaf_ids(go)))

    def test_insert_proxy_pull_leaves(self):
        dag = copy_dag(go_bp_main)
        original_size = len(dag)
        go_root = "GO:0000000"
        imbalanced = sum(is_imbalanced(term) for term in dag.values())
        dag_size = original_size
        leaf_ids = all_leaf_ids(dag)
        # Balancing iterations
        while imbalanced > 0:
            insert_proxy_terms(dag, dag[go_root], original_size)
            imbalanced = sum(is_imbalanced(term) for term in dag.values())
            print(f"Proxy terms added: {len(dag) - dag_size}")
            print(f"Imbalanced terms left: {imbalanced}")
            dag_size = len(dag)
        print("COMPLETED: Balancing DAG with proxies")

        print(f"\nNumber of original non-leaf terms: {original_size - len(leaf_ids)}")
        print(f"Number of leaves: {len(all_leaf_ids(dag))}")
        print(f"Number of inserted balancing proxies: {len(dag) - original_size}")
        average_leaf_depth = sum(dag[leaf_id].depth for leaf_id in all_leaf_ids(dag)) / len(all_leaf_ids(dag))
        print(f"Average leaf depth: {average_leaf_depth:.1f}\n")

        pull_leaves_down(dag, original_size)

    def test_merge_prune_pull_leaves(self):
        go = copy_dag(go_bp_main)
        # Plot leaf depth distribution before and after merge-prune convergence
        leaf_ids = all_leaf_ids(go)
        if plot:
            plot_depth_distribution(go, leaf_ids)
        # First merge, then prune:
        pruning_events = 1
        merge_events = 1
        parent_threshold = 1
        children_threshold = 1
        while pruning_events + merge_events > 0:
            merge_events = merge_chains(go, parent_threshold, children_threshold)
            pruning_events = prune_skip_connections(go)
        print("COMPLETED: Merge-prune until convergence")
        pull_leaves_down(go, len(go))

    def test_load_relationships(self):
        go_rel_full = GODag(OBO_PATH, optional_attrs={"relationship"})
        go_rel = filter_by_namespace(go_rel_full, {"biological_process"})
        del go_rel_full
        self.assertIsNotNone(go_rel["GO:0002893"].relationship)
        pass

    def test_leaf_relationships(self):
        go = copy_dag(go_bp_main)
        go_rel_full = copy_dag(go_rel_main)
        go_rel = filter_by_namespace(go_rel_full, {"biological_process"})
        del go_rel_full
        leaf_ids = all_leaf_ids(go)
        leaf_ids_with_rel = [leaf_id for leaf_id in leaf_ids if len(go_rel[leaf_id].relationship_rev) > 0]
        print(f"{len(leaf_ids_with_rel)}/{len(leaf_ids)} leaves have children through relationships.")

    def test_relationships_to_parents(self):
        go_bp_rel = filter_by_namespace(copy_dag(go_rel_main), {"biological_process"})
        relationships_to_parents(go_bp_rel)
        self.assertEqual(len(go_bp_rel["GO:0000027"].children), 3)
        self.assertEqual(len(go_bp_rel["GO:0000027"].parents), 3)

    def test_plot_relationship_effect(self):
        go_ref = copy_dag(go_main)
        go_rel_ref = copy_dag(go_rel_main)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        if plot:
            plot_depth_distribution(go_ref, go_ref.keys(), sub_fig=ax1,
                                    title="Only 'is_a' relationship")
            ax1.legend(["Original"], title="(#p, #c)")
            plot_depth_distribution(go_rel_ref, go_rel_ref.keys(), sub_fig=ax2,
                                    title="All relationships")
            ax2.legend(["Original"], title="(#p, #c)")
            ax2.set_ylabel("")
            fig.suptitle("Distribution of GO-term depths after Merge-Prune-Balance-Pull")
            plt.show()

        merge_conditions = [(0, 0), (1, 1), (1, 2), (1, 3), (2, 3), (2, 4)]
        for merge_condition in merge_conditions:
            print(f"\n----- Merge condition: {merge_condition} -----")
            go = copy_dag(go_ref)
            go_rel = copy_dag(go_rel_ref)
            merge_prune_until_convergence(go, merge_condition[0], merge_condition[1])
            merge_prune_until_convergence(go_rel, merge_condition[0], merge_condition[1])
            balance_until_convergence(go)
            balance_until_convergence(go_rel)
            pull_leaves_down(go, len(go_ref))
            pull_leaves_down(go_rel, len(go_rel_ref))
            print_dag_info(go)
            print_dag_info(go_rel)
            fig, (ax1, ax2) = plt.subplots(1, 2)
            if plot:
                plot_depth_distribution(go_ref, go_ref.keys(), sub_fig=ax1)
                plot_depth_distribution(go, go.keys(), sub_fig=ax1,
                                        title="Only 'is_a' relationship")
                plot_depth_distribution(go_rel_ref, go_rel_ref.keys(), sub_fig=ax2)
                plot_depth_distribution(go_rel, go_rel.keys(), sub_fig=ax2,
                                        title="All relationships")
                ax1.legend(["Original", merge_condition], title="(#p, #c)")
                ax2.legend(["Original", merge_condition], title="(#p, #c)")
                ax2.set_ylabel("")
                fig.suptitle("Distribution of GO-term depths after Merge-Prune-Balance-Pull")
                plt.show()

    def test_merge_prune_balance_pull_bp(self):
        go = copy_dag(go_bp_main)
        merge_prune_until_convergence(go, 1, 1)
        balance_until_convergence(go)
        pull_leaves_down(go, len(go_bp_main))
        print_dag_info(go)

    def test_merge_prune_balance_pull(self):
        go = copy_dag(go_main)
        merge_prune_until_convergence(go, 1, 1)
        balance_until_convergence(go)
        pull_leaves_down(go, len(go_bp_main))
        print_dag_info(go)

    def test_layerization_results(self):
        go = copy_dag(go_rel_main)
        balance_until_convergence(go)
        pull_leaves_down(go, len(go_rel_main))
        print_dag_info(go)
        layers = create_layers(go)
        self.assertIsNotNone(layers)

    def test_merge_condition_effects(self):
        merge_condition = (1, 30)
        go = copy_dag(go_rel_main)
        print_dag_info(go)
        plot_depth_distribution(go, go.keys(), alpha=0.2)

        merge_prune_until_convergence(go, merge_condition[0], merge_condition[1])

        balance_until_convergence(go)
        pull_leaves_down(go, len(go_rel_main))
        print_dag_info(go)
        if plot:
            plot_depth_distribution(go, go.keys())
            plt.legend(["(≥0, ≥0)", f"(≥{merge_condition[0]}, ≥{merge_condition[1]})"], title="(#parents, #children)")
            plt.show()

    def test_debug_merge_rel(self):
        """Out of 1081 merges in rel not in isa, 1060 were Case 1. The other 21 remain unidentified.
        To work, the 'Debug:' lines in merge_chains() and merge_prune_until_convergence() must be uncommented."""
        # go_ref = copy_dag(go_main)
        # go_rel_ref = copy_dag(go_rel_main)
        # go = copy_dag(go_main)
        # go_rel = copy_dag(go_rel_main)
        # go_merge1 = copy_dag(go_main)
        # go_rel_merge1 = copy_dag(go_rel_main)
        # go_prune = copy_dag(go_main)
        # go_rel_prune = copy_dag(go_rel_main)
        # go_merge2 = copy_dag(go_main)
        # go_rel_merge2 = copy_dag(go_rel_main)
        #
        # # All merge iterations, commented out for faster debugging
        # removed_terms_converged = merge_prune_until_convergence(go, 1, 1)
        # removed_terms_rel_converged = merge_prune_until_convergence(go_rel, 1, 1)
        # # iteration_difference_converged = []
        # # for i in range(len(removed_terms_converged)):
        # #     iteration_difference_converged.append(list(set(removed_terms_converged[i])-set(removed_terms_rel_converged[i])))
        #
        # _, removed_terms1 = merge_chains(go_merge1, 1, 1)
        # _, removed_terms_rel1 = merge_chains(go_rel_merge1, 1, 1)
        #
        # merge_chains(go_prune, 1, 1)
        # prune_skip_connections(go_prune)
        # merge_chains(go_rel_prune, 1, 1)
        # prune_skip_connections(go_rel_prune)
        #
        # merge_chains(go_merge2, 1, 1)
        # prune_skip_connections(go_merge2)
        # _, removed_terms2 = merge_chains(go_merge2, 1, 1)
        # merge_chains(go_rel_merge2, 1, 1)
        # prune_skip_connections(go_rel_merge2)
        # _, removed_terms_rel2 = merge_chains(go_rel_merge2, 1, 1)
        #
        # overlap_merge1 = list(set(removed_terms1).intersection(set(removed_terms_rel1)))
        # difference_isa_rel_merge1 = list(set(removed_terms1) - set(removed_terms_rel1))
        # difference_rel_isa_merge1 = list(set(removed_terms_rel1) - set(removed_terms1))
        # difference_rel_isa_converged = list(set(removed_terms_rel_converged) - set(removed_terms_converged))
        #
        # # overlap_m1 = [go_rel_main_parentless[x] for x in overlap_merge1]
        # # diff_i_r_m1 = [go_rel_main_parentless[x] for x in difference_isa_rel_merge1]
        # # diff_r_i_m1 = [go_rel_main_parentless[x] for x in difference_rel_isa_merge1]
        # diff_r_i_conv = [go_rel_main_parentless[x] for x in difference_rel_isa_converged]
        #
        # # Terms that are removed in rel, but not in isa:
        # # Are they all examples of Case 1? (leaf in isa, not in rel) -> Yes
        # case_1 = []
        # other = []
        # for i, term_id in enumerate(difference_rel_isa_converged):
        #     if len(go_ref[term_id].children) == 0:
        #         if len(go_rel_ref[term_id].children) > 0:
        #             case_1.append(diff_r_i_conv[i])
        #             continue
        #     other.append(diff_r_i_conv[i])
        # pass

    @requires_gaf
    def test_link_genes_bp(self):
        """Result: Three genes not linked because their annotated GO terms are obsolete."""
        go = copy_dag(go_bp_main)
        print_dag_info(go)
        link_genes_to_go_by_namespace(go, str(GAF_PATH), "biological_process")
        print_dag_info(go)
        print(f"Genes not on leafs: {len(genes_not_on_leaves_ids(go))}")

    @requires_gaf
    def test_link_genes_all_namespaces(self):
        """Result: Genes are added to non-leaf GO-terms."""
        go = copy_dag(go_rel_main)
        print_dag_info(go)
        link_genes_to_go_by_namespace(go, str(GAF_PATH), "biological_process")
        link_genes_to_go_by_namespace(go, str(GAF_PATH), "cellular_component")
        link_genes_to_go_by_namespace(go, str(GAF_PATH), "molecular_function")
        print_dag_info(go)
        print(f"Genes not on leafs: {len(genes_not_on_leaves_ids(go))}")

    @requires_gaf
    def test_link_merge_prune_balance_pull_bp(self):
        """Result: All genes with valid annotations are linked (36k)."""
        go = copy_dag(go_bp_main)
        print_dag_info(go)
        link_genes_to_go_by_namespace(go, str(GAF_PATH), "biological_process")
        merge_prune_until_convergence(go, 1, 10)
        balance_until_convergence(go)
        pull_leaves_down(go, len(go_bp_main))
        print_dag_info(go)
        print(f"Genes not on leafs: {len(genes_not_on_leaves_ids(go))}")
        layers = create_layers(go)
        pass

    @requires_gaf
    def test_unannotated_leaves(self):
        """Result: 36k gene-leaves, 20k term-leaves (57k leaves total)."""
        go = copy_dag(go_main)
        link_genes_to_go_by_namespace(go, str(GAF_PATH), "biological_process")
        leaf_ids = all_leaf_ids(go)
        non_gene_leaf_ids = [term_id for term_id in leaf_ids if not isinstance(go[term_id], GeneTerm)]
        gene_leaf_ids = [term_id for term_id in leaf_ids if isinstance(go[term_id], GeneTerm)]
        pass

    @requires_gaf
    def test_remove_geneless_branches_bp(self):
        """Result: Works :)"""
        go = copy_dag(go_bp_main)
        link_genes_to_go_by_namespace(go, str(GAF_PATH), "biological_process")
        print_dag_info(go)
        remove_geneless_branches(go)
        print_dag_info(go)

    @requires_gaf
    def test_full_cycle_bp(self):
        go = copy_dag(go_bp_main)
        link_genes_to_go_by_namespace(go, str(GAF_PATH), "biological_process")
        remove_geneless_branches(go)
        merge_prune_until_convergence(go, 1, 10)
        balance_until_convergence(go)
        pull_leaves_down(go, len(go_bp_main))
        layers = create_layers(go)
        print_dag_info(go)
        pass

    @requires_gaf
    def test_full_cycle_bp_dag_shape(self):
        go = copy_dag(go_bp_main)
        print_dag_info(go)
        print("Original GO")
        print_layers(create_layers(go))

        link_genes_to_go_by_namespace(go, str(GAF_PATH), "biological_process")
        print_layers(create_layers(go))

        remove_geneless_branches(go)
        print_layers(create_layers(go))

        merge_prune_until_convergence(go, 1, 10)
        print_layers(create_layers(go))

        balance_until_convergence(go)
        print_layers(create_layers(go))

        pull_leaves_down(go, len(go_bp_main))
        print_layers(create_layers(go))
        print_dag_info(go)

    def test_alternative_ids(self):
        """Result: keys() and values() are the same size, despite the presence of 2045 alt_ids.
        So values just contains duplicates, and it doesn't matter which one you use."""
        go = copy_dag(go_bp_main)
        alt_ids = [alt_id for alt_id in go.keys() if is_alternative_id(go, alt_id)]
        keys = go.keys()
        values = go.values()

    def test_create_layers2(self):
        go = copy_dag(go_bp_main)
        objs = create_layers(go)
        ori = create_layers_deprecated(go)

    @requires_gaf
    def test_save_gene_ids(self):
        go = copy_dag(go_bp_main)
        link_genes_to_go_by_namespace(go, str(GAF_PATH), "biological_process")
        # save_gene_ids(go, "../../GO_TCGA/gene_ids_in_go_bp_test.txt")

    @requires_gaf
    @requires_gene_ids
    def test_link_genes_from_subset(self):
        go = copy_dag(go_bp_main)
        genes = read_gene_ids(str(GENE_IDS_PATH))
        link_genes_to_go_by_namespace(go, str(GAF_PATH), "biological_process", genes)
        remove_geneless_branches(go)
        print_dag_info(go)
        pass
