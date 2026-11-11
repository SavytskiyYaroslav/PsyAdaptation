def test_imports():
    import psyadapt
    from psyadapt.reliability.base import BaseReliabilityEstimator
    from psyadapt.io.csv import load_csv_matrix

    assert psyadapt.__version__ == "0.1.0"
    assert BaseReliabilityEstimator is not None
    assert load_csv_matrix is not None
