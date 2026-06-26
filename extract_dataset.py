import synapseclient
import synapseutils
import pandas as pd
from pathlib import Path

# ── Configuración ─────────────────────────────────────────────────────────────
ENTITY_ID    = 'syn53972968'
DATASET_DIR  = Path(__file__).parent / 'dataset'
ELIGIBLE_CSV = Path(__file__).parent / 'outputs' / 'eligible_fc_subjects.csv'

TARGET_PATTERNS = ['_T1w.nii', '_task-rest_bold.nii']
# ──────────────────────────────────────────────────────────────────────────────

DATASET_DIR.mkdir(exist_ok=True)

syn = synapseclient.login(
    authToken='eyJ0eXAiOiJKV1QiLCJraWQiOiJXN05OOldMSlQ6SjVSSzpMN1RMOlQ3TDc6M1ZYNjpKRU9VOjY0NFI6VTNJWDo1S1oyOjdaQ0s6RlBUSCIsImFsZyI6IlJTMjU2In0.eyJhY2Nlc3MiOnsic2NvcGUiOlsidmlldyIsImRvd25sb2FkIiwibW9kaWZ5Il0sIm9pZGNfY2xhaW1zIjp7fX0sInRva2VuX3R5cGUiOiJQRVJTT05BTF9BQ0NFU1NfVE9LRU4iLCJpc3MiOiJodHRwczovL3JlcG8tcHJvZC5wcm9kLnNhZ2ViYXNlLm9yZy9hdXRoL3YxIiwiYXVkIjoiMCIsIm5iZiI6MTc4MTE4MzU3MywiaWF0IjoxNzgxMTgzNTczLCJqdGkiOiIzOTY3MCIsInN1YiI6IjM1OTEyNDQifQ.ED8TcRwhsYPDF5zaNpSmp59EgzGD4SoeadJ4hfq-QLKJVryn9xM4q4YsmMGPUQXuxTTQ_5dbhEb9tXMdDWRJm6SW9_PHkMg3BEfG8yF1kpZAhPY6Q25zOCjyumnwm4gfsm6s93PTd4cGOMaCDRJaka8sEYEReJDaedwjHI5F32-s0VLTO-dHnoUE1nrNIhZk8M9rKOcQbMbV_tEZUrsurBGUqYSGUrOrn4YR7CF5U9wda2mD8hOVPJ1jDpKwt0u0m_1tx34pVUAW-Czea-TbqBQYrR5ubHr929tjOh_ltAd_I00ucv8_7xC7i1OfXQOewmMBnK1uJw6abqqryz4JnQ',
    silent=True
)
print("Login OK")

eligible_ids = set(pd.read_csv(ELIGIBLE_CSV)['MRI_ID'].tolist())
print(f"Sujetos elegibles: {len(eligible_ids)}")
print(f"Destino:           {DATASET_DIR}\n")

downloaded = skipped = errors = 0

for dirpath, dirnames, filenames in synapseutils.walk(syn, ENTITY_ID):
    folder_str = dirpath[0]

    for filename, file_id in filenames:
        if not any(p in filename for p in TARGET_PATTERNS):
            skipped += 1
            continue

        is_eligible = (
            any(sub_id in folder_str for sub_id in eligible_ids) or
            any(sub_id in filename   for sub_id in eligible_ids)
        )
        if not is_eligible:
            skipped += 1
            continue

        local_dir  = DATASET_DIR / folder_str
        local_dir.mkdir(parents=True, exist_ok=True)
        local_file = local_dir / filename

        if local_file.exists():
            print(f"[YA EXISTE] {filename}")
            skipped += 1
            continue

        try:
            print(f"[↓] {folder_str}/{filename}")
            syn.get(file_id, downloadLocation=str(local_dir))
            downloaded += 1
        except Exception as e:
            print(f"[ERROR] {filename}: {e}")
            errors += 1

print(f"\n{'='*55}")
print(f"Descargados : {downloaded}")
print(f"Saltados    : {skipped}")
print(f"Errores     : {errors}")

if downloaded == 0 and errors == 0:
    print("\nNo se encontraron archivos con los filtros aplicados.")
    print("Revisá la estructura del entity en Synapse.")
