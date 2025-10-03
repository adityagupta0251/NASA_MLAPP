import os
from typing import Dict , List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # app/
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATHS: Dict[str, str] = {
    "rf": os.path.join(MODELS_DIR, "random_forest_model.pkl"),
    "xgb": os.path.join(MODELS_DIR, "xgb_model.pkl"),
    "lr": os.path.join(MODELS_DIR, "logistic_model.pkl")
}


# Feature columns matching training data
FEATURE_COLUMNS: List[str] = [
    "OrbitalPeriod_days",
    "OrbitalPeriodUpperUnc_days",
    "OrbitalPeriodLowerUnc_days",
    "TransitEpoch_BKJD",
    "TransitEpochUpperUnc_BKJD",
    "TransitEpochLowerUnc_BKJD",
    "ImpactParamete",
    "ImpactParameterUpperUnc",
    "ImpactParameterLowerUnc",
    "TransitDuration_hrs",
    "TransitDurationUpperUnc_hrs",
    "TransitDurationLowerUnc_hrs",
    "TransitDepth_ppm",
    "TransitDepthUpperUnc_ppm",
    "TransitDepthLowerUnc_ppm",
    "PlanetaryRadius_Earthradii",
    "PlanetaryRadiusUpperUnc_Earthradii",
    "PlanetaryRadiusLowerUnc_Earthradii",
    "EquilibriumTemperatureK",
    "EquilibriumTemperatureUpperUncK",
    "EquilibriumTemperatureLowerUncK",
    "InsolationFlux_Earthflux",
    "InsolationFluxUpperUnc_Earthflux",
    "InsolationFluxLowerUnc_Earthflux",
    "TransitSignal-to-Nois",
    "TCEPlanetNumbe",
    "StellarEffectiveTemperatureK",
    "StellarEffectiveTemperatureUpperUncK",
    "StellarEffectiveTemperatureLowerUncK",
    "StellarSurfaceGravity_log10(cm/s**2)",
    "StellarSurfaceGravityUpperUnc_log10(cm/s**2)",
    "StellarSurfaceGravityLowerUnc_log10(cm/s**2)",
    "StellarRadius_Solarradii",
    "StellarRadiusUpperUnc_Solarradii",
    "StellarRadiusLowerUnc_Solarradii",
    "RA_decimaldegrees",
    "Dec_decimaldegrees",
    "Kepler-band_mag",
    "DispositionScore"
]

CLASS_LABELS = {0: "FALSE_POSITIVE", 1: "CANDIDATE"}
