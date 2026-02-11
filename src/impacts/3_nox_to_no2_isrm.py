import os

import numpy as np
import pandas as pd

from impacts import functions


def main() -> None:
    mywd = "~/Dropbox/Research/SmartGrid_Behavioral/TransportationInitiative/ATLAS/BEAM_AQM/Rscripts"
    os.chdir(os.path.expanduser(mywd))

    rdatdir = os.path.join("..", "RData")

    res_data = functions.read_rdata(os.path.join(rdatdir, "SFB_NOX_NOX_ISRM.RData"))
    if "res" not in res_data:
        raise KeyError("Expected 'res' in SFB_NOX_NOX_ISRM.RData")
    res = res_data["res"].copy()

    no2ratio_data = functions.read_rdata(os.path.join(rdatdir, "sfb.no2ratio_isrmGRID.RData"))
    if "no2ratio" not in no2ratio_data:
        raise KeyError("Expected 'no2ratio' in sfb.no2ratio_isrmGRID.RData")
    no2ratio = no2ratio_data["no2ratio"].copy()

    res.index = pd.to_numeric(res.index, errors="coerce").astype(int)
    res.columns = [int(col) for col in res.columns]

    s_isrm = res.index.to_numpy()
    s_isrm = s_isrm[(s_isrm > 3) & (s_isrm != 3554)]

    res = res.loc[s_isrm, s_isrm]

    missing = no2ratio["isrm"].astype(int)
    if 843 not in missing.to_numpy():
        extra = pd.DataFrame({"isrm": [843], "NO2_NOx_ratio": [0.94]})
        no2ratio = pd.concat([no2ratio, extra], ignore_index=True)

    dat = res.transpose()
    dat["isrm"] = dat.index.astype(int)
    dat = dat.merge(no2ratio, on="isrm", how="left")

    cols = [col for col in dat.columns if isinstance(col, int) and 843 <= col <= 3706]
    dat[cols] = dat[cols].multiply(dat["NO2_NOx_ratio"], axis=0)

    res_dat = dat[cols].transpose()
    res_dat.columns = dat["isrm"].to_numpy()

    functions.write_rdata(os.path.join(rdatdir, "NOx_to_NO2_ISRM.RData"), {"res.dat": res_dat})

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    rng = np.random.default_rng()
    if res_dat.shape[0] >= 50:
        indices = rng.choice(res_dat.shape[0], size=50, replace=False)
        sample = res_dat.iloc[indices, indices]
        sample = sample.where(sample >= 0.001)
        plt.imshow(sample, aspect="auto")
        plt.show()


if __name__ == "__main__":
    main()
