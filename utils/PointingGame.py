import torch

class PointingGame:
    r"""Pointing game.
    Args:
        num_classes (int): number of classes in the dataset.
        tolerance (int, optional): tolerance (in pixels) of margin around
            ground truth annotation. Default: 15.
    Attributes:
        hits (:class:`torch.Tensor`): :attr:`num_classes`-dimensional vector of
            hits counts.
        misses (:class:`torch.Tensor`): :attr:`num_classes`-dimensional vector
            of misses counts.
    """

    def __init__(self, num_classes, tolerance=15):
        assert isinstance(num_classes, int)
        assert isinstance(tolerance, int)
        self.num_classes = num_classes
        self.tolerance = tolerance
        self.hits = torch.zeros((num_classes,), dtype=torch.float64)
        self.misses = torch.zeros((num_classes,), dtype=torch.float64)

    # def evaluate(self, mask, point):
    #     r"""Evaluate a point prediction.
    #     The function tests whether the prediction :attr:`point` is within a
    #     certain tolerance of the object ground-truth region :attr:`mask`
    #     expressed as a boolean occupancy map.
    #     Use the :func:`reset` method to clear all counters.
    #     Args:
    #         mask (:class:`torch.Tensor`): :math:`\{0,1\}^{H\times W}`.
    #         point (tuple of ints): predicted point :math:`(u,v)`.
    #     Returns:
    #         int: +1 if the point hits the object; otherwise -1.
    #     """
    #     # Get an acceptance region around the point. There is a hit whenever
    #     # the acceptance region collides with the class mask.
    #     v, u = torch.meshgrid((
    #         (torch.arange(mask.shape[0],
    #                       dtype=torch.float32) - point[1])**2,
    #         (torch.arange(mask.shape[1],
    #                       dtype=torch.float32) - point[0])**2,
    #     ))
    #     accept = (v + u) < self.tolerance**2

    #     # Test for a hit with the corresponding class.
    #     hit = (mask & accept).view(-1).any()

    #     return +1 if hit else -1

    def evaluate(self, bbox, point):
        r"""Evaluate a point prediction.
        The function tests whether the prediction :attr:`point` is within a
        certain tolerance of the object ground-truth region :attr:`mask`
        expressed as a boolean occupancy map.
        Use the :func:`reset` method to clear all counters.
        Args:
            bbox (tuples pf ints): the ground-truth bounding box :math: `(x_0, y_0, x_1, y_1)`
            point (tuple of ints): predicted point :math:`(u,v)`.
        Returns:
            int: +1 if the point hits the object; otherwise -1.
        """
        # Get an acceptance region around the point. There is a hit whenever
        # the acceptance region collides with the class mask.
        x0, y0, x1, y1 = bbox
        x0 = max(x0-self.tolerance, 0)
        x1 = x1 + self.tolerance
        y0 = max(y0-self.tolerance, 0)
        y1 = y1 + self.tolerance

        u, v = point
        if (u >= x0) and (u <= x1) and (v >= y0) and (v <= y1):
            hit = True
        else:
            hit = False

        return +1 if hit else -1

    def aggregate(self, hit, class_id):
        """Add pointing result from one example."""
        if hit == 0:
            return
        if hit == 1:
            self.hits[class_id] += 1
        elif hit == -1:
            self.misses[class_id] += 1
        else:
            assert False

    def reset(self):
        """Reset hits and misses."""
        self.hits = torch.zeros_like(self.hits)
        self.misses = torch.zeros_like(self.misses)

    @property
    def class_accuracies(self):
        """
        (:class:`torch.Tensor`): :attr:`num_classes`-dimensional vector
            containing per-class accuracy.
        """
        return self.hits / (self.hits + self.misses).clamp(min=1)

    @property
    def accuracy(self):
        """
        (:class:`torch.Tensor`): mean accuracy, computed by averaging
            :attr:`class_accuracies`.
        """
        return self.class_accuracies.mean()

    def __str__(self):
        class_accuracies = self.class_accuracies
        return '{:4.1f}% ['.format(100 * class_accuracies.mean()) + " ".join([
            '{}:{:4.1f}%'.format(c, 100 * a)
            for c, a in enumerate(class_accuracies)
        ]) + ']'
