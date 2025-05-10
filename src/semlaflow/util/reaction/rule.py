from dataclasses import dataclass
from omegaconf import DictConfig
from rdkit import Chem
from rdkit.Chem.rdChemReactions import ChemicalReaction, ReactionFromSmarts
from rdkit.Chem import Mol as RDMol


@dataclass
class RuleConfig:
    reaction: str
    compose: str
    decompose: str
    label: tuple[str, str]

    @classmethod
    def from_config(cls, cfg: DictConfig):
        return cls(cfg.reaction, cfg.compose, cfg.decompose, cfg.label)


class Rule:
    def __init__(self, config: DictConfig):
        self.config: RuleConfig = RuleConfig.from_config(config)
        self._compose_template: str = self.config.compose
        self._decompose_template: str = self.config.decompose
        self._rxn: ChemicalReaction = self.__init_reaction(self._compose_template)
        self._rev_rxn: ChemicalReaction = self.__init_reaction(self._decompose_template)
        assert self._rxn.GetNumReactantTemplates() == 2
        assert self._rev_rxn.GetNumReactantTemplates() == 1
        self.label: tuple[str, str] = (self.config.label[0], self.config.label[1])

        self.first_reactant_pattern: Chem.Mol = self._rxn.GetReactantTemplate(0)
        self.second_reactant_pattern: Chem.Mol = self._rxn.GetReactantTemplate(1)
        self.product_pattern: Chem.Mol = self._rev_rxn.GetReactantTemplate(0)

    def __init_reaction(self, template: str) -> ChemicalReaction:
        """Initializes a reaction by converting the SMARTS-pattern to an `rdkit` object."""
        rxn = ReactionFromSmarts(template)
        ChemicalReaction.Initialize(rxn)
        return rxn

    def is_reactant(self, mol: RDMol, is_first: bool = True) -> bool:
        """Checks if a molecule is the reactant for the reaction."""
        if is_first:
            return mol.HasSubstructMatch(self.first_reactant_pattern)
        else:
            return mol.HasSubstructMatch(self.second_reactant_pattern)

    def is_product(self, mol: RDMol) -> bool:
        """Checks if a molecule is a reactant for the reaction."""
        return mol.HasSubstructMatch(self.product_pattern)

    def forward(self, reactant1: RDMol, reactant2: RDMol) -> RDMol | None:
        """Runs the reaction on a set of reactants and returns the product.

        Args:
            reactant1: First reactant
            reactant2: Second reactant

        Returns:
            The product of the reaction or `None` if the reaction is not possible.
        """

        # Run reaction
        ps: list[list[RDMol]] = self._rxn.RunReactants((reactant1, reactant2), 10)
        if len(ps) == 0:
            raise ValueError("Reaction did not yield any products.")
        p = ps[0][0]
        try:
            Chem.SanitizeMol(p)
            p = Chem.RemoveHs(p)
            return p
        except (Chem.rdchem.KekulizeException, Chem.rdchem.AtomValenceException):
            return None

    def reverse(self, product: RDMol) -> list[tuple[RDMol, RDMol]]:
        """Runs the reverse reaction on a product, to return the reactants.

        Args:
            product: A tuple of RDMol object of the product (now reactant) to run the reverse reaction on.

        Returns:
            The possible reactant pairs of the reaction.
        """
        try:
            rs_list: list[list[RDMol]] = self._rev_rxn.RunReactants((product,), 10)
        except Exception:
            return []

        res: list[tuple[RDMol, RDMol]] = []
        for rs in rs_list:
            if len(rs) != 2:
                continue
            r1, r2 = self.__refine_molecule(rs[0]), self.__refine_molecule(rs[1])
            if r1 is None or r2 is None:
                continue
            res.append((r1, r2))
        return res

    @staticmethod
    def __refine_molecule(mol: RDMol) -> RDMol | None:
        smi = Chem.MolToSmiles(mol)
        if "[CH]" in smi:
            smi = smi.replace("[CH]", "C")
        return Chem.MolFromSmiles(smi)
