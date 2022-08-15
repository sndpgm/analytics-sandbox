import pytest

from sandbox.features.features import FeaturesManager


class TestFeatures:
    def test__check_column_values(self):
        column_values = ["date"]
        assert FeaturesManager._check_column_values(column_values) == column_values

    def test__raise_check_column_values(self):
        column_values = "date"
        msg = "The value of `column_values` must be list."
        with pytest.raises(ValueError, match=msg):
            FeaturesManager._check_column_values(column_values)

    def test__check_column_id(self):
        column_id = ["category_id", "subcategory_id"]
        assert FeaturesManager._check_column_id(column_id) == column_id

    def test__raise_check_column_id(self):
        column_id = "category_id"
        msg = "The value of `column_id` must be list."
        with pytest.raises(ValueError, match=msg):
            FeaturesManager._check_column_id(column_id)

    def test__check_sort_values(self):
        sort_values = ["date"]
        assert FeaturesManager._check_sort_values(sort_values) == sort_values

    def test__raise_check_sort_values(self):
        sort_values = "date"
        msg = "The value of `sort_values` must be list."
        with pytest.raises(ValueError, match=msg):
            FeaturesManager._check_sort_values(sort_values)

    def test__check_func_param_list(self):
        ...
