"""
Tests for Advanced Feature Transformer Pipeline
"""

import pytest
import pandas as pd
import numpy as np
from sklearn_pandas.feature_transformer_pipeline import (
    AdvancedTransformerPipeline,
    create_feature_pipeline,
)
from sklearn.preprocessing import StandardScaler


class TestAdvancedTransformerPipeline:
    """Test suite for AdvancedTransformerPipeline"""
    
    def test_init(self):
        """Test initialization of AdvancedTransformerPipeline"""
        transformers = [("scaler", StandardScaler())]
        pipeline = AdvancedTransformerPipeline(transformers=transformers)
        
        assert pipeline.transformers == transformers
        assert pipeline.feature_names is None
