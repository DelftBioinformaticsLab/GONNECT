from goatools.obo_parser import GOTerm


class DAGGenerator:
    """Class for small DAG examples."""

    #       A
    #     / |
    #   /   B
    #  | /    \
    #  C       D
    @staticmethod
    def dag1():
        """Single skip connection. Used for copy, prune, insert proxy and update level and depth tests."""
        a = GOTerm("biological_process")
        b = GOTerm("biological_process")
        c = GOTerm("biological_process")
        d = GOTerm("biological_process")
        a.item_id = "A"
        b.item_id = "B"
        c.item_id = "C"
        d.item_id = "D"
        dag = {"A": a, "B": b, "C": c, "D": d}

        a.children = {b, c}
        b.parents = {a}
        b.children = {c, d}
        c.parents = {a, b}
        d.parents = {b}
        a.depth, b.depth, c.depth, d.depth = 0, 1, 2, 2
        a.level, b.level, c.level, d.level = 0, 1, 1, 2

        return dag

    #      A
    #    / |
    #  /   B
    # | /
    # C
    @staticmethod
    def dag2():
        """Single skip connection or single (1, 1) merge. Used for merge test."""
        a = GOTerm("biological_process")
        b = GOTerm("biological_process")
        c = GOTerm("biological_process")
        a.item_id = "A"
        b.item_id = "B"
        c.item_id = "C"
        dag = {"A": a, "B": b, "C": c}

        a.children = {b, c}
        b.parents = {a}
        b.children = {c}
        c.parents = {a, b}
        a.depth, b.depth, c.depth = 0, 1, 2
        a.level, b.level, c.level = 0, 1, 1

        return dag

    #       A
    #     / |
    #   /   B
    #  | /    \
    #  C       D
    #           \
    #            E
    @staticmethod
    def dag3():
        """Single skip connection and single (1, 1) merge. Used for pull test."""
        a, b, c, d, e = (GOTerm("biological_process") for _ in range(5))
        a.item_id = "A"
        b.item_id = "B"
        c.item_id = "C"
        d.item_id = "D"
        e.item_id = "E"
        dag = {"A": a, "B": b, "C": c, "D": d, "E": e}

        a.children = {b, c}
        b.parents = {a}
        b.children = {c, d}
        c.parents = {a, b}
        d.parents = {b}
        d.children = {e}
        e.parents = {d}
        a.depth, b.depth, c.depth, d.depth, e.depth = 0, 1, 2, 2, 3
        a.level, b.level, c.level, d.level, e.level = 0, 1, 1, 2, 3

        return dag

    #       A
    #     / |
    #   /   B
    #  | /    \
    #  C       D
    #         / \
    #        E   F
    @staticmethod
    def dag4():
        """Single balance proxy and single pull proxy. Used for training loop test."""
        a, b, c, d, e, f = (GOTerm("biological_process") for _ in range(6))
        a.item_id = "A"
        b.item_id = "B"
        c.item_id = "C"
        d.item_id = "D"
        e.item_id = "E"
        f.item_id = "F"
        dag = {"A": a, "B": b, "C": c, "D": d, "E": e, "F": f}

        a.children = {b, c}
        b.parents = {a}
        b.children = {c, d}
        c.parents = {a, b}
        d.parents = {b}
        d.children = {e, f}
        e.parents = {d}
        f.parents = {d}
        a.depth, b.depth, c.depth, d.depth, e.depth, f.depth = 0, 1, 2, 2, 3, 3
        a.level, b.level, c.level, d.level, e.level, f.level = 0, 1, 1, 2, 3, 3

        return dag
