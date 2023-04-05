from tensors import Tensors
import numpy as np


class CtmAlg:
    def __init__(self, bond, transfer_bond):
        self.tensors = Tensors()
        self.bond = bond
        self.transfer_bond = transfer_bond
        self.corner_tensor = self.tensors.random_tensor(bond, bond)
        self.edge_tensor = self.tensors.random_tensor(bond, bond, transfer_bond)
        self.a_tensor = self.tensors.lattice_tensor_a()

    def eval_corner(self) -> np.array:
        pass
