import re
import qcportal
import qcelemental as qcel
import numpy as np
import pandas as pd
from pprint import pprint as pp
from qcportal.singlepoint import QCSpecification

client = qcportal.PortalClient(
    "http://localhost:7777",
    verify=False,
)

def queue_manybody(
    mol: qcel.models.Molecule = None,
    use_CP: bool = None,
    method: str = None,
    basis: str = None,
):
    cp_str = "cp" if use_CP else "nocp"
    qc_spec = QCSpecification(
        program="psi4",
        driver="energy",
        method=method,
        basis=basis,
    )
    _, ids = client.add_manybodys(
        [mol],
        program="qcmanybody",
        levels={
            1: qc_spec,
            2: qc_spec,
        },
        bsse_correction=[cp_str],
        # tag='phoenix-agent',
        tag='free',
        keywords={},
    )
    return ids

def queue_manybodys(df: pd.DataFrame = None):
    manybody_ids_list = []
    for _, row in df.iterrows():
        method = row["Level of Theory"].split("/")[0]
        basis = row["Level of Theory"].split("/")[1]
        cp_str = row["Level of Theory"].split("/")[2]
        use_CP = False if "un" in cp_str else True
        manybody_ids = queue_manybody(
            mol=row["qcel_dimer"],
            use_CP=use_CP,
            method=method,
            basis=basis,
        )
        manybody_ids_list.append(manybody_ids)
    df["qcfractal id"] = manybody_ids_list
    df["job status"] = np.nan
    df["mb_interaction_energy"] = np.nan
    df["mb_wall_time"] = np.nan
    df["psi4 output"] = None
    return df

def _normalize_id(val):
    if isinstance(val, list):
        return val[0] if val else None
    return val

def _bsse_from_level(level_str: str) -> str:
    cp_str = level_str.split("/")[2].lower()
    return "nocp" if ("nocp" in cp_str or "un" in cp_str) else "cp"

def _extract_interaction_energy(rec, bsse: str):
    props = rec.properties or {}
    if "interaction_energy" in props:
        return props["interaction_energy"]
    if "interaction_energies" in props:
        ie = props["interaction_energies"]
        if isinstance(ie, dict):
            if bsse in ie:
                return ie[bsse]
            if len(ie) == 1:
                return next(iter(ie.values()))
    for k in ("cp_interaction_energy", "nocp_interaction_energy",
              "interaction_energy_cp", "interaction_energy_nocp"):
        if k in props:
            return props[k]
    return np.nan

def _report_manybody_errors(rec):
    print(f"manybody id {rec.id} status {rec.status}")
    print("manybody error:", rec.error)
    errs = rec.children_errors
    print("children errors:", [e.id for e in errs])

    for e in errs:
        print("child id:", e.id, "status:", e.status)
        print("child error:", e.error)
        print("child stderr:", e.stderr)
        print("child stdout:", e.stdout)

def retrieve_manybodies(df: pd.DataFrame = None):
    df = df.copy()
    ids = [_normalize_id(x) for x in df["qcfractal id"].tolist()]
    records = client.get_manybodys(ids, include=["clusters", "**"])

    for idx, rec in enumerate(records):
        if rec is None:
            df.at[idx, "job status"] = "RecordStatusEnum.error"
            continue
        df.at[idx, "job status"] = str(rec.status)
        if rec.status == "error":
            _report_manybody_errors(rec)
            df.at[idx, "mb_interaction_energy"] = np.nan
            df.at[idx, "mb_wall_time"] = np.nan
            df.at[idx, "psi4 output"] = None
            continue

        bsse = _bsse_from_level(df.at[idx, "Level of Theory"])
        df.at[idx, "mb_interaction_energy"] = _extract_interaction_energy(rec, bsse)
        wall_time = np.nan

        if rec.compute_history and rec.compute_history[-1].provenance:
            # wall_time = rec.compute_history[-1].provenance.wall_time
            print(rec.compute_history[-1].provenance)
        df.at[idx, "mb_wall_time"] = wall_time
        outputs = []

        for c in rec.clusters or []:
            sp = c.singlepoint_record
            if sp is None:
                continue
            outputs.append({
                "singlepoint_id": sp.id,
                "mc_level": c.mc_level,
                "fragments": c.fragments,
                "basis": c.basis,
                "stdout": sp.stdout,
                "stderr": sp.stderr,
            })

        df.at[idx, "psi4 output"] = outputs

    return df

def soft_delete_manybodies_with_children(df: pd.DataFrame):
    ids = [_normalize_id(x) for x in df["qcfractal id"].tolist()]
    ids = [i for i in ids if i is not None]
    if not ids:
        return None
    meta = client.delete_records(ids, soft_delete=True, delete_children=True)
    print("manybody delete meta:", meta)
    return meta

def cancel_then_soft_delete_manybodies(df: pd.DataFrame):
    ids = [_normalize_id(x) for x in df["qcfractal id"].tolist()]
    ids = [i for i in ids if i is not None]
    if not ids:
        return None
    meta_cancel = client.cancel_records(ids)
    print("cancel meta:", meta_cancel)
    meta_delete = client.delete_records(ids, soft_delete=True, delete_children=True)
    print("delete meta:", meta_delete)
    return meta_delete

def check_manybody_progress(df: pd.DataFrame, max_print=5):
    ids = [_normalize_id(x) for x in df["qcfractal id"].tolist()]
    ids = [i for i in ids if i is not None]
    recs = client.get_manybodys(ids, include=["clusters", "**"])
    for r in recs[:max_print]:
        print(f"\nmanybody id {r.id} status {r.status}")
        if r.status in ("waiting", "running"):
            try:
                print("waiting reason:", client.get_waiting_reason(r.id))
            except Exception as e:
                print("waiting reason error:", e)
        # child statuses
        if r.clusters:
            child_statuses = {}
            for c in r.clusters:
                sp = c.singlepoint_record
                if sp is None:
                    continue
                child_statuses.setdefault(str(sp.status), 0)
                child_statuses[str(sp.status)] += 1
            print("child status counts:", child_statuses)
        # print last stdout/error if complete/error
        if r.status == "error":
            print("manybody error:", r.error)
        if r.status == "complete":
            print("manybody stdout (tail):")
            out = r.stdout or ""
            print(out[-1000:])

def main():
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    df = pd.read_pickle("../../../src/qcml_mcp/select-LoT/runs/local-only-test-2/select_lot_df.pkl")
    pp(df)
    # df1 = queue_manybodys(df)
    # pp(df1)
    # # Later, after records run
    # df2 = retrieve_manybodies(df1)
    # pp(df2)

    # # If user asks about status
    # check_manybody_progress(df2)

    # # #   DANGER ZONE!   # # #
    
    # # If records error out and you have adressed the error
    # soft_delete_manybodies_with_children(df2)

    # # If records are running and you need to cancel/delete them
    # cancel_then_soft_delete_manybodies(df2)

    return
    
if __name__ == "__main__":
    main()
