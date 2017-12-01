import unittest

from statoil.utils import batches
import numpy as np


class TestUtils(unittest.TestCase):

    def test_batch1(self):
        x = [[1], [2], [3]]
        y = ['a', 'b', 'c']
        x_batches_correct = [([1], [2]), ([3], )]
        y_batches_correct = [('a', 'b'), ('c', )]
        x_batches, y_batches = [], []
        for batch in batches(list(zip(x, y)), batch_size=2, shuffle=False):
            x_batch, y_batch = zip(*batch)
            x_batches.append(x_batch)
            y_batches.append(y_batch)

        self.assertEqual(x_batches, x_batches_correct)
        self.assertEqual(y_batches, y_batches_correct)

    def test_batch2(self):
        x = np.array([[1, 2], [3, 4], [5, 6]])
        batches_correct = [[[1, 2], [3, 4]], [[5, 6]]]
        for i, batch in enumerate(batches(x, batch_size=2, shuffle=False)):
            self.assertEqual(batches_correct[i], batch.tolist())



if __name__ == '__main__':
    unittest.main()