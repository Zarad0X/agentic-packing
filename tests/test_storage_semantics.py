import unittest

from physcensis.assets import PrimitiveAssetCatalog
from physcensis.storage_semantics import semantically_compatible_support
from physcensis.types import SceneObject


class StorageSemanticsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = PrimitiveAssetCatalog()

    def _object(self, category: str) -> SceneObject:
        return SceneObject(
            f"{category}_0",
            self.catalog.resolve_category(category, f"{category}_0"),
        )

    def test_fragile_electronics_are_not_generic_shelves(self) -> None:
        self.assertFalse(
            semantically_compatible_support(
                self._object("book"),
                [self._object("laptop")],
            )
        )

    def test_paper_family_and_loose_fillers_are_compatible(self) -> None:
        book = self._object("book")
        self.assertTrue(
            semantically_compatible_support(self._object("notebook"), [book])
        )
        self.assertTrue(semantically_compatible_support(self._object("pen"), [book]))


if __name__ == "__main__":
    unittest.main()
