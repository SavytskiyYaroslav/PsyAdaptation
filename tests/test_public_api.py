def test_public_api_imports():
    import psyadapt

    assert hasattr(psyadapt, "compare_reliability_csv")
    assert hasattr(psyadapt, "compare_reliability_to_excel")
    assert hasattr(psyadapt, "workbook_vertical_to_matrix")
    assert hasattr(psyadapt, "workbook_vertical_to_csv")
