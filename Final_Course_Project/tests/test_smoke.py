#!/usr/bin/env python3
"""
Smoke tests for the Workshop 4 simulation pipeline.

Tests that all scripts can be imported and basic functions work.
"""

import sys
import logging
from pathlib import Path

# Setup
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data_processing"))

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all main modules can be imported."""
    logger.info("Testing imports...")
    
    try:
        import pandas as pd
        import numpy as np
        import joblib
        import matplotlib.pyplot as plt
        from scipy.ndimage import label, convolve
        import sklearn
        logger.info("✓ All core dependencies imported successfully")
        return True
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False

def test_data_files():
    """Test that required data files exist."""
    logger.info("Testing data files...")
    
    required_files = [
        ROOT / "data" / "raw" / "PersonalityData_ExternalVersion001.csv",
        ROOT / "data" / "psychopathy_DOOM_DATA.csv" if (ROOT / "data" / "psychopathy_DOOM_DATA.csv").exists() else None,
    ]
    
    all_exist = True
    for f in required_files:
        if f is None:
            logger.warning("⚠ DOOM DATA not generated yet (run generate_doom_data.py first)")
        elif f.exists():
            logger.info(f"✓ {f.name} exists")
        else:
            logger.error(f"✗ Missing: {f}")
            all_exist = False
    
    return all_exist or Path(ROOT / "data" / "raw" / "PersonalityData_ExternalVersion001.csv").exists()

def test_model_file():
    """Test that trained model exists."""
    logger.info("Testing model file...")
    
    model_path = ROOT / "results" / "rf_final.pkl"
    
    if model_path.exists():
        logger.info(f"✓ Model file exists: {model_path.name}")
        return True
    else:
        logger.warning(f"⚠ Model not trained yet (run train_and_save_model.py first)")
        return False

def test_output_directories():
    """Test that output directories can be created."""
    logger.info("Testing output directories...")
    
    output_dirs = [
        ROOT / "results" / "ml",
        ROOT / "results" / "ca",
        ROOT / "results" / "tail_metrics",
        ROOT / "results" / "analysis",
    ]
    
    try:
        for d in output_dirs:
            d.mkdir(parents=True, exist_ok=True)
            if d.exists():
                logger.info(f"✓ {d.name} directory ready")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to create directories: {e}")
        return False

def test_numpy_operations():
    """Test basic NumPy operations used in simulations."""
    logger.info("Testing NumPy operations...")
    
    try:
        import numpy as np
        from scipy.ndimage import convolve
        
        # Test grid creation and convolution
        grid = np.random.rand(40, 40)
        kernel = np.ones((3, 3)) / 9.0
        result = convolve(grid, kernel, mode='constant', cval=0)
        
        assert result.shape == grid.shape, "Convolution shape mismatch"
        logger.info("✓ NumPy operations (convolution) working")
        
        # Test clipping
        clipped = np.clip(grid, 0, 1)
        assert clipped.min() >= 0 and clipped.max() <= 1, "Clipping failed"
        logger.info("✓ NumPy clipping operations working")
        
        return True
    except Exception as e:
        logger.error(f"✗ NumPy operations failed: {e}")
        return False

def test_pandas_operations():
    """Test basic Pandas operations used in simulations."""
    logger.info("Testing Pandas operations...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Test DataFrame operations
        df = pd.DataFrame({
            'a': np.random.rand(100),
            'b': np.random.rand(100),
            'c': np.random.rand(100)
        })
        
        # Test numeric selection and fillna
        numeric_df = df.select_dtypes(include=[np.number]).fillna(-1)
        assert numeric_df.shape[1] == 3, "select_dtypes failed"
        logger.info("✓ Pandas select_dtypes and fillna working")
        
        # Test CSV writing/reading
        test_csv = ROOT / "results" / ".test_pandas.csv"
        df.to_csv(test_csv, index=False)
        df_read = pd.read_csv(test_csv)
        assert df_read.shape == df.shape, "CSV read/write mismatch"
        test_csv.unlink()  # cleanup
        logger.info("✓ Pandas CSV operations working")
        
        return True
    except Exception as e:
        logger.error(f"✗ Pandas operations failed: {e}")
        return False

def test_scikit_learn_operations():
    """Test basic scikit-learn operations."""
    logger.info("Testing scikit-learn operations...")
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        import numpy as np
        
        # Create dummy data
        X = np.random.rand(100, 10)
        y = np.random.rand(100)
        
        # Train simple model
        model = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=-1)
        model.fit(X, y)
        
        # Make predictions
        preds = model.predict(X)
        
        # Compute metrics
        mse = mean_squared_error(y, preds)
        mae = mean_absolute_error(y, preds)
        
        assert mse > 0 and mae > 0, "Metrics invalid"
        logger.info(f"✓ scikit-learn operations working (MSE={mse:.4f}, MAE={mae:.4f})")
        
        return True
    except Exception as e:
        logger.error(f"✗ scikit-learn operations failed: {e}")
        return False

def run_all_tests():
    """Run all smoke tests."""
    logger.info("=" * 60)
    logger.info("Starting Workshop 4 Smoke Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Data Files", test_data_files),
        ("Model File", test_model_file),
        ("Output Directories", test_output_directories),
        ("NumPy Operations", test_numpy_operations),
        ("Pandas Operations", test_pandas_operations),
        ("scikit-learn Operations", test_scikit_learn_operations),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            logger.error(f"✗ {name} test crashed: {e}")
            results[name] = False
        logger.info("")
    
    # Summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {name}")
    
    logger.info("=" * 60)
    logger.info(f"Result: {passed}/{total} tests passed")
    logger.info("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
