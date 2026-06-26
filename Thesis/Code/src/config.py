from dataclasses import dataclass
from pathlib import Path

# Raíz del repositorio (dos niveles arriba de este archivo: src/ → Code/ → Thesis/)
_THESIS_ROOT = Path(__file__).resolve().parents[2]
_DATA        = _THESIS_ROOT.parent / "data-iipsi" / "data"

# Rutas listas para usar — no requieren edición manual
DEFAULT_PATHS = None  # se inicializa abajo para que _THESIS_ROOT esté disponible


@dataclass(frozen=True)
class Paths:
    excel_path: Path
    fc_folder: Path
    t1w_csv_path: Path
    out_dir: Path


DEFAULT_PATHS = Paths(
    excel_path   = _DATA / "matrices-redlat" / "datos-redlat.xlsx",
    fc_folder    = _DATA / "matrices-redlat" / "matrices-redlat",
    t1w_csv_path = _DATA / "Redlat_VGM_AAL_.csv",
    out_dir      = _THESIS_ROOT / "Outputs",
)


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    diagnoses_to_use: tuple[str, ...] = ("CN", "AD", "FTD")
    test_size: float = 0.10
    k_folds: int = 5
    fisher_z: bool = True
    use_optuna: bool = True
    optuna_xgb_trials: int = 100
    optuna_vae_trials: int = 60
    reuse_artifacts: bool = True
