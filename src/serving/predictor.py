"""CoffeePredictor: loads all trained artifacts at app startup, runs inference per request."""

import logging
import pickle
import sys
from pathlib import Path
from typing import Optional

import polars as pl

# Add src/ and project root to sys.path so sibling packages resolve
_SRC = Path(__file__).parent.parent
_ROOT = _SRC.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT))

from features.feature_manager import CoffeeFeatureManager
from features.feature_selector import LassoFeatureSelector
from features.tfidf_extractor import TfidfExtractor
from features.topic_extractor import TopicExtractor
from pipeline.constants import EXCLUDE_COLUMNS

logger = logging.getLogger(__name__)

MODELS_DIR = _ROOT / "models"

# Columns that are never model features — strip them before selector.transform()
_EXCLUDE = set(EXCLUDE_COLUMNS) | {"desc_1", "desc_2", "desc_3"}


class CoffeePredictor:
    """Loads trained artifacts at startup, runs inference on requests."""

    def __init__(self) -> None:
        self.fm: Optional[CoffeeFeatureManager] = None
        self.selector: Optional[LassoFeatureSelector] = None
        self.model = None
        self._loaded = False

    def load(self) -> None:
        """Load all artifacts. Called once from FastAPI lifespan on startup.

        BERT loading takes ~1-2 minutes — this is expected.
        """
        logger.info("Loading serving artifacts from %s", MODELS_DIR)

        # 1. TF-IDF extractor
        tfidf = TfidfExtractor({"models_dir": str(MODELS_DIR)})
        tfidf.load_extractor(
            str(MODELS_DIR)
        )  # method is load_extractor, NOT load_vectorizer
        logger.info("TF-IDF vectorizer loaded")

        # 2. Topic extractor (LDA + NMF)
        topics = TopicExtractor({"models_dir": str(MODELS_DIR)})
        topics.load_models(str(MODELS_DIR))
        logger.info("Topic models loaded")

        # 3. Feature manager — inject pre-loaded extractors directly
        self.fm = CoffeeFeatureManager(
            {
                "extractors": {
                    "tfidf": True,
                    "bert": True,  # pre-trained DistilBERT — no pkl to load
                    "topics": True,
                    "sentiment": True,  # pre-trained DistilBERT sentiment — no pkl to load
                    "glove": True,  # vectors cached at ~/gensim-data/glove-wiki-gigaword-300/
                }
            }
        )
        self.fm.extractors["tfidf"] = tfidf
        self.fm.extractors["topics"] = topics
        # bert and sentiment extractors were already created by CoffeeFeatureManager.__init__
        self.fm.is_fitted = True
        logger.info("Feature manager ready (BERT/sentiment loaded as pre-trained)")

        # 4. LASSO feature selector
        selector_path = MODELS_DIR / "lasso_feature_selector.pkl"
        self.selector = LassoFeatureSelector.load_selector(selector_path)
        logger.info("LASSO selector loaded")

        # 5. Regression model
        model_path = MODELS_DIR / "xgboost_model.pkl"
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        logger.info("XGBoost model loaded — predictor ready")

        self._loaded = True

    def predict(
        self,
        desc_1: str,
        desc_2: str = "",
        desc_3: str = "",
        roast: str = "",
        country_of_origin: str = "",
        roaster: str = "",
        aroma: Optional[float] = None,
        acid: Optional[float] = None,
        body: Optional[float] = None,
        flavor: Optional[float] = None,
        aftertaste: Optional[float] = None,
    ) -> float:
        """Run inference on a single request."""
        if not self._loaded:
            raise RuntimeError("CoffeePredictor.load() must be called before predict()")

        # Build single-row Polars DataFrame matching training data schema.
        # Categorical columns must be present (even if empty) so the encoder
        # runs and produces its full feature set — unknown values map to
        # 'Other' / all-zeros per the encoder's handle_unknown config.
        row = {
            "desc_1": [desc_1],
            "desc_2": [desc_2 or ""],
            "desc_3": [desc_3 or ""],
            "roast": [roast or ""],
            "country_of_origin": [country_of_origin or ""],
            "roaster": [roaster or ""],
        }
        df = pl.DataFrame(row)

        # Extract all features — same pipeline as training
        features_df = self.fm.extract_all_features(
            df, text_columns=["desc_1", "desc_2", "desc_3"]
        )

        # Drop non-feature columns, convert to pandas for sklearn
        feature_cols = [c for c in features_df.columns if c not in _EXCLUDE]
        X = features_df.select(feature_cols).to_pandas()

        # Apply LASSO selection (reduces to ~279 features)
        X_selected = self.selector.transform(X)

        # Predict (model.predict returns np.ndarray)
        return float(self.model.predict(X_selected)[0])

    @property
    def is_loaded(self) -> bool:
        """Check if predictor is ready."""
        return self._loaded
