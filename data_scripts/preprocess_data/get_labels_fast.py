#!/usr/bin/env python
# get_labels_fast.py
#
# Build a zero-copy post-discharge label cache:
#   • labels_scalar.bin   (N×2 uint8   →   memory-map)
#   • hadm_row_map.pkl    {hadm_id: row}
#   • phecode_current.pkl {hadm_id: tuple(PheCodes)}
#   • phecode_next.pkl    {hadm_id: tuple(PheCodes at next adm)}
#   • label_matrix.feather   (optional analytics table)

import argparse, os, pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather
from tqdm import tqdm


def get_phecode_df(base_path: str, cache_dir: Path):
    # Check if phecode mappings already exist
    phecode_mappings_path = os.path.join(cache_dir, "phecode_mappings.pkl")
    if os.path.exists(phecode_mappings_path):
        # Load existing mappings
        mappings = pickle.load(open(phecode_mappings_path, "rb"))
        phecode_to_idx = mappings['phecode_to_idx']
        idx_to_phecode = mappings['idx_to_phecode']
        phe_code_size = mappings['phe_code_size']
        print(f"Loaded PHE code mappings with {phe_code_size} unique codes")
        
        # Also load the phecode dataframe if needed
        phecode_df = None
        if os.path.exists(os.path.join(cache_dir, "phecode_df.pkl")):
            phecode_df = pickle.load(open(os.path.join(cache_dir, "phecode_df.pkl"), "rb"))
        pickle.dump(phe_code_size, open(os.path.join(cache_dir, "phecode_size.pkl"), "wb"))
        return phecode_df, phecode_to_idx, idx_to_phecode, phe_code_size

    if os.path.exists(os.path.join(cache_dir, "phecode_df.pkl")):
        phecode_df = pickle.load(open(os.path.join(cache_dir, "phecode_df.pkl"), "rb"))
        
        # Create mappings from the existing dataframe
        unique_phecodes = sorted(phecode_df['PheCode'].unique())
        phecode_to_idx = {code: idx for idx, code in enumerate(unique_phecodes)}
        idx_to_phecode = {idx: code for idx, code in enumerate(unique_phecodes)}
        phe_code_size = len(unique_phecodes)
        
        # Save the mappings
        mappings = {
            'phecode_to_idx': phecode_to_idx,
            'idx_to_phecode': idx_to_phecode,
            'phe_code_size': phe_code_size
        }
        pickle.dump(mappings, open(phecode_mappings_path, "wb"))
        pickle.dump(phe_code_size, open(os.path.join(cache_dir, "phecode_size.pkl"), "wb"))
        print(f"Created and saved PHE code mappings with {phe_code_size} unique codes")
        return phecode_df, phecode_to_idx, idx_to_phecode, phe_code_size
        
    # If phecode dataframe doesn't exist, create it
    print("Phecode labels not found in cache. Computing...")

    icd_to_phe_mapping = pd.read_csv(base_path + 'icd_to_phecode.csv')
    diagnoses_df = pd.read_csv(base_path + 'diagnoses_icd.csv')
    diagnoses_names_df = pd.read_csv(base_path + 'd_icd_diagnoses.csv')
    icd_to_phe_mapping.columns = ['icd_code','PheCode','icd_version']
    icd_to_phe_mapping['icd_version'] = icd_to_phe_mapping['icd_version'].str.replace('ICD', '')

    diagnoses_df['icd_version'] = diagnoses_df['icd_version'].astype(str)
    icd_to_phe_mapping['icd_version'] = icd_to_phe_mapping['icd_version'].astype(str)
    diagnoses_phecode = diagnoses_df.merge(icd_to_phe_mapping, on=['icd_code', 'icd_version'], how='left')

    diagnoses_names_df['icd_version'] = diagnoses_names_df['icd_version'].astype(str)
    diagnoses_names_df['icd_code'] = diagnoses_names_df['icd_code'].astype(str)
    diagnoses_names_df['icd_version'] = diagnoses_names_df['icd_version'].astype(str)
    diagnoses_phecode_names = diagnoses_phecode.merge(diagnoses_names_df, on=['icd_code', 'icd_version'], how='left')

    diagnoses_phecode_names['Rollup_Status'] = diagnoses_phecode_names['PheCode'].notna().replace({True: '1', False: '0'})
    diagnoses_phecode_names_filtered = diagnoses_phecode_names[diagnoses_phecode_names['Rollup_Status'] == '1']

    # Save the phecode DataFrame
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    pickle.dump(diagnoses_phecode_names_filtered, open(os.path.join(cache_dir, "phecode_df.pkl"), "wb"))
    phecode_df = diagnoses_phecode_names_filtered
    
    # Create and save the mappings
    unique_phecodes = sorted(phecode_df['PheCode'].unique())
    phecode_to_idx = {code: idx for idx, code in enumerate(unique_phecodes)}
    idx_to_phecode = {idx: code for idx, code in enumerate(unique_phecodes)}
    phe_code_size = len(unique_phecodes)
    
    mappings = {
        'phecode_to_idx': phecode_to_idx,
        'idx_to_phecode': idx_to_phecode,
        'phe_code_size': phe_code_size
    }
    pickle.dump(mappings, open(phecode_mappings_path, "wb"))
    print(f"Created and saved PHE code mappings with {phe_code_size} unique codes")
    pickle.dump(phe_code_size, open(os.path.join(cache_dir, "phecode_size.pkl"), "wb"))
    
    # Index phecode_df by hadm_id for faster lookups
    if phecode_df.index.name != 'hadm_id':
        print("Indexing phecode_df by hadm_id...")
        phecode_df.set_index('hadm_id', inplace=True)
        print("Indexing complete.")
    
    return phecode_df, phecode_to_idx, idx_to_phecode, phe_code_size

# ────────────────────────────────────────────────────────────────────
def build_label_cache(cache_dir: Path, out_dir: Path, max_phe_codes: int = 20, verbose: bool = True) -> None:
    """
    Vectorise mortality / readmission + PheCode targets for *all* admissions,
    then write one memory-mapped binary and four tiny pickles.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    adm_path = cache_dir / "merged_with_disch_df_final_filtered.pkl"
    phe_path = cache_dir / "phecode_df.pkl"

    if verbose: print(f"→ loading admissions   : {adm_path}")
    adm = pd.read_pickle(adm_path)
    adm = adm.sort_values(["hadm_id", "admittime"]).drop_duplicates("hadm_id", keep="last")

    if verbose: print(f"→ loading PheCode DataFrame : {phe_path}")
    phe = pd.read_pickle(phe_path)[["hadm_id", "PheCode"]].dropna()

    # ----------------------------------------------------------------------
    # normalise datetime dtypes
    # ----------------------------------------------------------------------
    for c in ("admittime", "dischtime", "dod"):
        adm[c] = pd.to_datetime(adm[c], errors="coerce")

    # ----------------------------------------------------------------------
    # mortality ≤ 180 days
    # ----------------------------------------------------------------------
    delta_dod = adm["dod"] - adm["dischtime"]
    adm["mortality_6m"] = (
        adm["dod"].notna()
        & (delta_dod > pd.Timedelta(0))
        & (delta_dod <= pd.Timedelta(days=180))
    ).astype("uint8")

    # ----------------------------------------------------------------------
    # readmission ≤ 15 days
    # ----------------------------------------------------------------------
    adm = adm.sort_values(["subject_id", "admittime"])
    adm["next_hadm_id"]   = adm.groupby("subject_id")["hadm_id"].shift(-1)
    adm["next_admittime"] = adm.groupby("subject_id")["admittime"].shift(-1)

    delta_next = adm["next_admittime"] - adm["dischtime"]
    adm["readmission_15d"] = (
        adm["next_admittime"].notna()
        & (delta_next > pd.Timedelta(0))
        & (delta_next <= pd.Timedelta(days=15))
    ).astype("uint8")

    # ----------------------------------------------------------------------
    # PheCode dicts
    # ----------------------------------------------------------------------
    if verbose: print("→ building PheCode maps …")

    current_map = (
        phe.groupby("hadm_id")["PheCode"]
        .apply(lambda s: tuple(pd.Series(s).unique()))
        .to_dict()
    )

    next_map: dict[int, tuple] = {}
    for h, nh in zip(adm["hadm_id"].values, adm["next_hadm_id"].values):
        if pd.isna(nh):
            next_map[int(h)] = tuple()
        else:
            next_map[int(h)] = current_map.get(int(nh), tuple())

    mappings = pickle.load(open(cache_dir / "phecode_mappings.pkl", "rb"))
    phecode_to_idx = mappings["phecode_to_idx"]
    phe_code_size = mappings["phe_code_size"]

    current_idx = {h: tuple(phecode_to_idx[c] for c in codes)
                for h, codes in current_map.items()}

    next_idx = {h: tuple(phecode_to_idx[c] for c in codes)
                for h, codes in next_map.items()}
    # ─── Build padded index matrices for GPU gather ───
    # max_phe_codes = K, phe_code_size = P, number of adms = N
    K = max_phe_codes
    P = phe_code_size
    adm_ids = adm["hadm_id"].astype(int).tolist()
    N = len(adm_ids)

    # preallocate
    current_idx_mat = np.full((N, K), P, dtype=np.int32)
    current_len     = np.zeros((N,), dtype=np.int32)
    next_idx_mat    = np.full((N, K), P, dtype=np.int32)
    next_len        = np.zeros((N,), dtype=np.int32)

    # fill row-by-row in the admission order
    for i, h in enumerate(adm_ids):
        ci = current_idx.get(h, ())
        L = min(len(ci), K)
        if L:
            current_idx_mat[i, :L] = ci[:L]
        current_len[i] = L

        ni = next_idx.get(h, ())
        L2 = min(len(ni), K)
        if L2:
            next_idx_mat[i, :L2] = ni[:L2]
        next_len[i] = L2

    # Save numpy arrays for memory mapping
    np.save(out_dir / "current_idx_mat.npy", current_idx_mat)
    np.save(out_dir / "current_len.npy",     current_len)
    np.save(out_dir / "next_idx_mat.npy",    next_idx_mat)
    np.save(out_dir / "next_len.npy",        next_len)
    # ----------------------------------------------------------------------
    # write memory-mapped scalar matrix (mortality, readmission)
    # ----------------------------------------------------------------------
    scalar_mat = adm[["mortality_6m", "readmission_15d"]].to_numpy(
        dtype=np.uint8, copy=False
    )
    bin_path = out_dir / "labels_scalar.bin"
    if verbose: print(f"→ writing uint8 matrix → {bin_path}  ({scalar_mat.shape[0]} rows)")
    scalar_mat.tofile(bin_path)

    # row-index map
    with open(out_dir / "hadm_row_map.pkl", "wb") as f:
        pickle.dump({int(h): i for i, h in enumerate(adm["hadm_id"].tolist())},
                    f, protocol=pickle.HIGHEST_PROTOCOL)

    # PheCode pickles
    with open(out_dir / "phecode_current.pkl", "wb") as f:
        pickle.dump(current_map, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out_dir / "phecode_next.pkl", "wb") as f:
        pickle.dump(next_map, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(out_dir / "phecode_current_idx.pkl", "wb") as f:
        pickle.dump(current_idx, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out_dir / "phecode_next_idx.pkl", "wb") as f:
        pickle.dump(next_idx, f, protocol=pickle.HIGHEST_PROTOCOL)

    # optional Arrow table for sanity checks
    feather.write_feather(
        adm[["hadm_id", "mortality_6m", "readmission_15d"]].reset_index(drop=True),
        out_dir / "label_matrix.feather",
    )

    if verbose:
        print("✔ label cache complete")
        print(f"   scalars  : {bin_path}")
        print(f"   row map  : {out_dir/'hadm_row_map.pkl'}")
        print(f"   phecodes : {out_dir/'phecode_current.pkl'}, …_next.pkl")
        print(f"   padded idx mats: shapes {current_idx_mat.shape}, {next_idx_mat.shape}")


# ───────────────────────────── CLI ────────────────────────────────
if __name__ == "__main__":
    ROOT_DIR = Path.cwd()
    temp_dfs_dir = ROOT_DIR / "temp_dfs_lite"
    p = argparse.ArgumentParser(
        description="Pre-compute post-discharge labels into a memory-mapped cache."
    )
    p.add_argument("--cache_dir", default=temp_dfs_dir,
                   help="Folder that already contains merged_with_disch_df* and phecode_df.pkl")
    p.add_argument("--out_dir",  default=temp_dfs_dir / "label_cache",
                   help="Where to put labels_scalar.bin + pickles")
    p.add_argument("--quiet", action="store_true", help="Suppress prints")
    args = p.parse_args()

    get_phecode_df(
        base_path='/Users/riccardoconci/Local_documents/!!MIMIC/hosp/',
        cache_dir=Path(args.cache_dir).resolve()
    )

    MAX_PHE_CODES = 20
    
    build_label_cache(
        cache_dir=Path(args.cache_dir).resolve(),
        out_dir=Path(args.out_dir).resolve(),
        verbose=not args.quiet,
        max_phe_codes=MAX_PHE_CODES
    )