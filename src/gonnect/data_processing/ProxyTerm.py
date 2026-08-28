from goatools.obo_parser import GOTerm


class ProxyTerm(GOTerm):
    """GOTerm subclass used as proxy for real GOTerm. Proxies are needed for balancing and ensuring branch lengths from root to gene are equal."""

    def __init__(self, item_id: str, parents: set[GOTerm], children: set[GOTerm]):
        super().__init__()
        self.item_id = item_id
        self.parents = parents
        self.children = children
        self.level = 0
        self.depth = 0
        self.namespace = "biological_process"

        # Update parent-child relationships of ProxyTerm neighbors
        for parent in self.parents:
            parent.children.add(self)
            parent.children -= self.children
        for child in self.children:
            child.parents.add(self)
            child.parents -= self.parents
