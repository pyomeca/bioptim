# import pytest
#
# import numpy as np
# from bioptim import Mapping, BiMapping, SelectionMapping, Dependency
#
#
# def test_mapping():
#     obj_to_map = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
#
#     np.testing.assert_almost_equal(Mapping([0, 2]).map(obj_to_map), [[0, 1, 2], [6, 7, 8]])
#     np.testing.assert_almost_equal(Mapping([None, 2, None]).map(obj_to_map), [[0, 0, 0], [6, 7, 8], [0, 0, 0]])
#     np.testing.assert_almost_equal(
#         Mapping([None, 2, 1], oppose=[1, 2]).map(obj_to_map), [[0, 0, 0], [-6, -7, -8], [-3, -4, -5]]
#     )
#     np.testing.assert_almost_equal(Mapping([None, 0], oppose=1).map(obj_to_map), [[0, 0, 0], [0, -1, -2]])
#
#
# def test_bidirectional_mapping():
#     mapping = BiMapping([0, 1, 2], [3, 4, 5])
#
#     np.testing.assert_almost_equal(len(mapping.to_first), 3)
#     np.testing.assert_almost_equal(mapping.to_first.map_idx, [3, 4, 5])
#     np.testing.assert_almost_equal(mapping.to_second.map_idx, [0, 1, 2])
#     np.testing.assert_almost_equal(mapping.to_second.map_idx, [0, 1, 2])
#
#     mapping_with_oppose = BiMapping([0, 1, 2], [3, 4, 5], 1, [1, 2])
#     np.testing.assert_almost_equal(mapping_with_oppose.to_second.map_idx, [0, 1, 2])
#     np.testing.assert_almost_equal(mapping_with_oppose.to_second.oppose, [1, -1, 1])
#     np.testing.assert_almost_equal(mapping_with_oppose.to_first.map_idx, [3, 4, 5])
#     np.testing.assert_almost_equal(mapping_with_oppose.to_first.oppose, [1, -1, -1])
#
#     with pytest.raises(RuntimeError, match="to_second must be a Mapping class"):
#         BiMapping(1, [3, 4, 5])
#     with pytest.raises(RuntimeError, match="to_first must be a Mapping class"):
#         BiMapping([0, 1, 2], 3)
#
#
# def test_bidirectional_selection_mapping():
#     mapping = SelectionMapping(5, (0, 1, 3), (Dependency(4, 3),))
#
#     assert len(mapping.to_first.map_idx) == 3
#     assert mapping.to_first.map_idx == [0, 1, 3]
#     assert len(mapping.to_second.map_idx) == 5
#     assert mapping.to_second.map_idx == [0, 1, None, 2, 2]
#
#     mapping_with_oppose = SelectionMapping(6, (3, 4, 5), (Dependency(4, 3, -1),))
#     assert mapping_with_oppose.to_second.map_idx == [None, None, None, 0, 0, 2]
#     assert mapping_with_oppose.to_second.oppose == [1, 1, 1, 1, -1, 1]
#     assert mapping_with_oppose.to_first.map_idx == [3, 4, 5]
#
#     with pytest.raises(ValueError, match="independent_indices must not contain more elements than nb_elements"):
#         SelectionMapping(1, (3, 4, 5))
#     with pytest.raises(ValueError, match="nb_dof should be an 'int'"):
#         SelectionMapping((0, 1, 2), 3)
