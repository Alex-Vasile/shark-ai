# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for default ops wrapper that preserves DefaultPrimitiveTensor type."""

import unittest
import torch
from sharktank import ops
from sharktank.types import DefaultPrimitiveTensor


class DefaultOpsWrapperTest(unittest.TestCase):
    """Test that default ops preserve DefaultPrimitiveTensor type with mixed inputs."""

    def test_cat_all_torch_tensors_returns_torch_tensor(self):
        """When all inputs are torch.Tensor, result should be torch.Tensor."""
        t1 = torch.rand(2, 3, dtype=torch.float32)
        t2 = torch.rand(2, 3, dtype=torch.float32)
        result = ops.cat([t1, t2], dim=0)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertNotIsInstance(result, DefaultPrimitiveTensor)
        self.assertEqual(tuple(result.shape), (4, 3))

    def test_cat_all_primitive_tensors_returns_default_primitive_tensor(self):
        """When all inputs are DefaultPrimitiveTensor, result should be DefaultPrimitiveTensor."""
        t1 = DefaultPrimitiveTensor(data=torch.rand(2, 3, dtype=torch.float32))
        t2 = DefaultPrimitiveTensor(data=torch.rand(2, 3, dtype=torch.float32))
        result = ops.cat([t1, t2], dim=0)
        
        self.assertIsInstance(result, DefaultPrimitiveTensor)
        self.assertEqual(tuple(result.shape), (4, 3))

    def test_cat_mixed_primitive_and_torch_returns_default_primitive_tensor(self):
        """When inputs are mixed PrimitiveTensor and torch.Tensor, result should be DefaultPrimitiveTensor."""
        t1 = DefaultPrimitiveTensor(data=torch.rand(2, 3, dtype=torch.float32))
        t2 = torch.rand(2, 3, dtype=torch.float32)
        result = ops.cat([t1, t2], dim=0)
        
        self.assertIsInstance(result, DefaultPrimitiveTensor)
        self.assertEqual(tuple(result.shape), (4, 3))

    def test_cat_mixed_torch_and_primitive_returns_default_primitive_tensor(self):
        """When inputs are mixed torch.Tensor and PrimitiveTensor, result should be DefaultPrimitiveTensor."""
        t1 = torch.rand(2, 3, dtype=torch.float32)
        t2 = DefaultPrimitiveTensor(data=torch.rand(2, 3, dtype=torch.float32))
        result = ops.cat([t1, t2], dim=0)
        
        self.assertIsInstance(result, DefaultPrimitiveTensor)
        self.assertEqual(tuple(result.shape), (4, 3))

    def test_matmul_all_torch_tensors_returns_torch_tensor(self):
        """matmul with all torch.Tensor inputs should return torch.Tensor."""
        t1 = torch.rand(3, 4, dtype=torch.float32)
        t2 = torch.rand(4, 5, dtype=torch.float32)
        result = ops.matmul(t1, t2)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertNotIsInstance(result, DefaultPrimitiveTensor)
        self.assertEqual(tuple(result.shape), (3, 5))

    def test_matmul_lhs_primitive_tensor_returns_default_primitive_tensor(self):
        """matmul with DefaultPrimitiveTensor LHS should return DefaultPrimitiveTensor."""
        t1 = DefaultPrimitiveTensor(data=torch.rand(3, 4, dtype=torch.float32))
        t2 = torch.rand(4, 5, dtype=torch.float32)
        result = ops.matmul(t1, t2)
        
        self.assertIsInstance(result, DefaultPrimitiveTensor)
        self.assertEqual(tuple(result.shape), (3, 5))

    def test_matmul_rhs_primitive_tensor_returns_default_primitive_tensor(self):
        """matmul with DefaultPrimitiveTensor RHS should return DefaultPrimitiveTensor."""
        t1 = torch.rand(3, 4, dtype=torch.float32)
        t2 = DefaultPrimitiveTensor(data=torch.rand(4, 5, dtype=torch.float32))
        result = ops.matmul(t1, t2)
        
        self.assertIsInstance(result, DefaultPrimitiveTensor)
        self.assertEqual(tuple(result.shape), (3, 5))

    def test_matmul_both_primitive_tensors_returns_default_primitive_tensor(self):
        """matmul with both DefaultPrimitiveTensor inputs should return DefaultPrimitiveTensor."""
        t1 = DefaultPrimitiveTensor(data=torch.rand(3, 4, dtype=torch.float32))
        t2 = DefaultPrimitiveTensor(data=torch.rand(4, 5, dtype=torch.float32))
        result = ops.matmul(t1, t2)
        
        self.assertIsInstance(result, DefaultPrimitiveTensor)
        self.assertEqual(tuple(result.shape), (3, 5))

    def test_elementwise_all_torch_tensors_returns_torch_tensor(self):
        """elementwise op with all torch.Tensor inputs should return torch.Tensor."""
        t1 = torch.rand(3, 4, dtype=torch.float32)
        t2 = torch.rand(3, 4, dtype=torch.float32)
        result = ops.elementwise(lambda x, y: x + y, t1, t2)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertNotIsInstance(result, DefaultPrimitiveTensor)
        self.assertEqual(tuple(result.shape), (3, 4))

    def test_elementwise_mixed_returns_default_primitive_tensor(self):
        """elementwise op with mixed inputs should return DefaultPrimitiveTensor."""
        t1 = DefaultPrimitiveTensor(data=torch.rand(3, 4, dtype=torch.float32))
        t2 = torch.rand(3, 4, dtype=torch.float32)
        result = ops.elementwise(lambda x, y: x + y, t1, t2)
        
        self.assertIsInstance(result, DefaultPrimitiveTensor)
        self.assertEqual(tuple(result.shape), (3, 4))


if __name__ == "__main__":
    unittest.main()
