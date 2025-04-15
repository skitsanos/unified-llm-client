"""
Test simple import functionality

@author: skitsanos
"""

import unittest


class TestImport(unittest.TestCase):
    """Test that the package can be imported"""

    def test_import(self):
        """Test that the package can be imported"""
        try:
            # Just verify import works
            import llm  # noqa

            self.assertTrue(True)
        except ImportError:
            self.fail("Failed to import from llm")


if __name__ == "__main__":
    unittest.main()
