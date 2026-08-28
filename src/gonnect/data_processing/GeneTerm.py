from goatools.obo_parser import GOTerm


class GeneTerm(GOTerm):

    def __init__(self, item_id: str, parents: set[GOTerm], namespace: str="biological_process"):
        super().__init__()
        self.item_id = item_id
        self.parents = parents
        self.children = set()
        self.level = 0
        self.depth = 0
        self.namespace = namespace

        for parent in self.parents:
            parent.children.add(self)
