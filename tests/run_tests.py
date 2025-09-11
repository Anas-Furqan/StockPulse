import unittest
import os
import sys

# Add the app directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import test modules
from test_data_fetcher import TestDataFetcher
from test_models import TestModelFactory, TestIsolationForestModel, TestLOFModel
from test_preprocessing import TestFeatureEngineering

def run_all_tests():
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestDataFetcher))
    test_suite.addTest(unittest.makeSuite(TestModelFactory))
    test_suite.addTest(unittest.makeSuite(TestIsolationForestModel))
    test_suite.addTest(unittest.makeSuite(TestLOFModel))
    test_suite.addTest(unittest.makeSuite(TestFeatureEngineering))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)