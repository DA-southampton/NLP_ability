import torch.nn as nn
import torch
import torch.cuda


class MatrixTree(nn.Module):
    """Implementation of the matrix-tree theorem for computing marginals
    of non-projective dependency parsing. This attention layer is used
    in the paper "Learning Structured Text Representations"
    :cite:`DBLP:journals/corr/LiuL17d`.
    """

    def __init__(self, eps=1e-5):
        self.eps = eps
        super(MatrixTree, self).__init__()

    def forward(self, input):
        laplacian = input.exp() + self.eps
        output = input.clone()
        for b in range(input.size(0)):
            lap = laplacian[b].masked_fill(
                torch.eye(input.size(1), device=input.device).ne(0), 0)
            lap = -lap + torch.diag(lap.sum(0))
            # store roots on diagonal
            lap[0] = input[b].diag().exp()
            inv_laplacian = lap.inverse()

            factor = inv_laplacian.diag().unsqueeze(1)\
                                         .expand_as(input[b]).transpose(0, 1)
            term1 = input[b].exp().mul(factor).clone()
            term2 = input[b].exp().mul(inv_laplacian.transpose(0, 1)).clone()
            term1[:, 0] = 0
            term2[0] = 0
            output[b] = term1 - term2
            roots_output = input[b].diag().exp().mul(
                inv_laplacian.transpose(0, 1)[0])
            output[b] = output[b] + torch.diag(roots_output)
        return output
