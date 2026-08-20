# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import unittest.mock

import torch

from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher


class TestPermute(unittest.TestCase):
    """Test AllToAllTokenDispatcher._permute which reorders tokens from rank-major to expert-major layout.

    Input layout:  (e0,r0), (e1,r0), ..., (e0,r1), (e1,r1), ...  (rank-major)
    Output layout: (e0,r0), (e0,r1), ..., (e1,r0), (e1,r1), ...  (expert-major)
    """

    def _make_dispatcher(self, num_ranks: int) -> AllToAllTokenDispatcher:
        """Create a minimal AllToAllTokenDispatcher for testing _permute."""
        cfg = AllToAllTokenDispatcher.Config(num_experts=1, top_k=1)
        dispatcher = AllToAllTokenDispatcher(cfg)
        # Mock ep_mesh with a simple object that has .size() returning num_ranks
        mock_mesh = unittest.mock.MagicMock()
        mock_mesh.size.return_value = num_ranks
        dispatcher.ep_mesh = mock_mesh
        return dispatcher

    def _permute(self, tokens_per_expert_group, experts_per_rank, num_ranks):
        """Helper that calls _permute and returns (permuted_indices, num_tokens_per_expert)."""
        dispatcher = self._make_dispatcher(num_ranks)
        total = tokens_per_expert_group.sum().item()
        dummy_input = torch.zeros(total, 1)
        _, _, permuted_indices, num_tokens_per_expert = dispatcher._permute(
            dummy_input, tokens_per_expert_group
        )
        return permuted_indices, num_tokens_per_expert

    def test_basic_2ranks_2experts(self):
        # 2 ranks, 2 experts per rank
        # tokens_per_expert_group: [r0e0, r0e1, r1e0, r1e1] = [2, 3, 1, 4]
        tokens_per_expert_group = torch.tensor([2, 3, 1, 4])
        permuted_indices, num_tokens_per_expert = self._permute(
            tokens_per_expert_group, experts_per_rank=2, num_ranks=2
        )

        # Expert-major layout: e0r0(2), e0r1(1), e1r0(3), e1r1(4)
        # Input positions:
        #   r0e0: [0, 1], r0e1: [2, 3, 4], r1e0: [5], r1e1: [6, 7, 8, 9]
        # Output order:
        #   e0r0: [0, 1], e0r1: [5], e1r0: [2, 3, 4], e1r1: [6, 7, 8, 9]
        expected_indices = torch.tensor([0, 1, 5, 2, 3, 4, 6, 7, 8, 9])
        torch.testing.assert_close(permuted_indices, expected_indices)

        # num_tokens_per_expert: sum across ranks for each expert
        # e0: r0e0 + r1e0 = 2 + 1 = 3, e1: r0e1 + r1e1 = 3 + 4 = 7
        expected_num_tokens = torch.tensor([3, 7])
        torch.testing.assert_close(num_tokens_per_expert, expected_num_tokens)

    def test_single_rank(self):
        # 1 rank, 3 experts: no reordering needed
        tokens_per_expert_group = torch.tensor([4, 2, 5])
        permuted_indices, num_tokens_per_expert = self._permute(
            tokens_per_expert_group, experts_per_rank=3, num_ranks=1
        )

        expected_indices = torch.arange(11)
        torch.testing.assert_close(permuted_indices, expected_indices)
        torch.testing.assert_close(num_tokens_per_expert, tokens_per_expert_group)

    def test_single_expert(self):
        # 3 ranks, 1 expert per rank: no reordering needed
        tokens_per_expert_group = torch.tensor([3, 5, 2])
        permuted_indices, num_tokens_per_expert = self._permute(
            tokens_per_expert_group, experts_per_rank=1, num_ranks=3
        )

        expected_indices = torch.arange(10)
        torch.testing.assert_close(permuted_indices, expected_indices)

        # Single expert gets all tokens
        expected_num_tokens = torch.tensor([10])
        torch.testing.assert_close(num_tokens_per_expert, expected_num_tokens)

    def test_zero_tokens_for_some_experts(self):
        # 2 ranks, 2 experts, some with zero tokens
        # [r0e0, r0e1, r1e0, r1e1] = [0, 3, 2, 0]
        tokens_per_expert_group = torch.tensor([0, 3, 2, 0])
        permuted_indices, num_tokens_per_expert = self._permute(
            tokens_per_expert_group, experts_per_rank=2, num_ranks=2
        )

        # Expert-major: e0r0(0), e0r1(2), e1r0(3), e1r1(0)
        # Input positions: r0e0: [], r0e1: [0, 1, 2], r1e0: [3, 4], r1e1: []
        # Output order: e0r0: [], e0r1: [3, 4], e1r0: [0, 1, 2], e1r1: []
        expected_indices = torch.tensor([3, 4, 0, 1, 2])
        torch.testing.assert_close(permuted_indices, expected_indices)

        expected_num_tokens = torch.tensor([2, 3])
        torch.testing.assert_close(num_tokens_per_expert, expected_num_tokens)

    def test_all_zero_tokens(self):
        tokens_per_expert_group = torch.tensor([0, 0, 0, 0])
        permuted_indices, num_tokens_per_expert = self._permute(
            tokens_per_expert_group, experts_per_rank=2, num_ranks=2
        )

        self.assertEqual(permuted_indices.numel(), 0)
        expected_num_tokens = torch.tensor([0, 0])
        torch.testing.assert_close(num_tokens_per_expert, expected_num_tokens)

    def test_uniform_distribution(self):
        # 3 ranks, 2 experts, uniform token counts
        # [r0e0, r0e1, r1e0, r1e1, r2e0, r2e1] = [2, 2, 2, 2, 2, 2]
        tokens_per_expert_group = torch.tensor([2, 2, 2, 2, 2, 2])
        permuted_indices, num_tokens_per_expert = self._permute(
            tokens_per_expert_group, experts_per_rank=2, num_ranks=3
        )

        # Expert-major: e0r0(2), e0r1(2), e0r2(2), e1r0(2), e1r1(2), e1r2(2)
        # Input positions:
        #   r0e0: [0,1], r0e1: [2,3], r1e0: [4,5], r1e1: [6,7], r2e0: [8,9], r2e1: [10,11]
        # Output: e0r0[0,1], e0r1[4,5], e0r2[8,9], e1r0[2,3], e1r1[6,7], e1r2[10,11]
        expected_indices = torch.tensor([0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 10, 11])
        torch.testing.assert_close(permuted_indices, expected_indices)

        expected_num_tokens = torch.tensor([6, 6])
        torch.testing.assert_close(num_tokens_per_expert, expected_num_tokens)

    def test_permutation_is_valid(self):
        # The output should be a permutation of [0, total)
        tokens_per_expert_group = torch.tensor([3, 1, 4, 1, 5, 9])
        permuted_indices, _ = self._permute(
            tokens_per_expert_group, experts_per_rank=3, num_ranks=2
        )

        total = tokens_per_expert_group.sum().item()
        self.assertEqual(permuted_indices.numel(), total)
        self.assertEqual(
            set(permuted_indices.tolist()),
            set(range(total)),
        )


def _torchao_triton_row_permute_available() -> bool:
    """Whether torchao provides the Triton row-permutation ops the parity
    test needs (skip cleanly on machines with old or absent torchao)."""
    try:
        from torchtitan.models.common.token_dispatcher import (
            _torchao_triton_row_permute_available as probe,
        )
    except ImportError:
        return False
    return probe()


class TestTorchAOTokenDispatcherRowPermute(unittest.TestCase):
    """TorchAOTokenDispatcher's padded permute: Triton row-permute op vs the
    eager advanced-indexing fallback must match bitwise, forward and backward
    (a row permutation is a pure copy)."""

    def _make_dispatcher(self, num_ranks: int, use_triton: bool):
        from torchtitan.models.common.token_dispatcher import TorchAOTokenDispatcher

        cfg = TorchAOTokenDispatcher.Config(
            num_experts=8,
            top_k=1,
            pad_multiple=32,
            use_triton_row_permute=use_triton,
        )
        dispatcher = TorchAOTokenDispatcher(cfg)
        mock_mesh = unittest.mock.MagicMock()
        mock_mesh.size.return_value = num_ranks
        dispatcher.ep_mesh = mock_mesh
        return dispatcher

    def test_config_defaults_to_triton(self):
        from torchtitan.models.common.token_dispatcher import TorchAOTokenDispatcher

        cfg = TorchAOTokenDispatcher.Config(num_experts=8, top_k=1, pad_multiple=32)
        self.assertTrue(cfg.use_triton_row_permute)

    @unittest.skipUnless(
        torch.cuda.is_available(), "requires CUDA (torchao Triton ops)"
    )
    @unittest.skipUnless(
        _torchao_triton_row_permute_available(),
        "requires torchao with the Triton row-permutation ops "
        "(permute_and_pad(use_triton=...) / triton_unpermute)",
    )
    def test_triton_matches_eager_bitwise(self):
        torch.manual_seed(42)
        # 2 ranks x 4 local experts, uneven counts incl. an empty expert.
        counts_E = torch.tensor(
            [40, 0, 17, 71, 5, 33, 60, 30], dtype=torch.int32, device="cuda"
        )
        num_tokens = int(counts_E.sum())
        dim = 256
        x_RD = torch.randn(
            num_tokens, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        x_eager_RD = x_RD.detach().clone().requires_grad_(True)
        grad_out = None
        results = {}
        for use_triton, x_in in ((True, x_RD), (False, x_eager_RD)):
            dispatcher = self._make_dispatcher(2, use_triton)
            permute_out = dispatcher._permute(x_in, counts_E)
            input_shape, permuted_RD, permuted_indices, counts_padded_e = permute_out
            unpermuted_RD = dispatcher._unpermute(
                permuted_RD, input_shape, permuted_indices
            )
            # Random cotangent, shared by both arms so the bitwise backward
            # parity actually discriminates permutation-index bugs (an all-ones
            # cotangent is invariant under any valid-index permutation).
            if grad_out is None:
                grad_out = torch.randn_like(unpermuted_RD)
            unpermuted_RD.backward(grad_out, retain_graph=False)
            results[use_triton] = (
                input_shape,
                permuted_RD.detach(),
                permuted_indices,
                counts_padded_e,
                unpermuted_RD.detach(),
            )
        for a, b in zip(results[True], results[False]):
            if isinstance(a, torch.Tensor):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            else:
                self.assertEqual(a, b)
        torch.testing.assert_close(x_RD.grad, x_eager_RD.grad, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
