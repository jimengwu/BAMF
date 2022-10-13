import pandas as pd
import numpy as np
from analysis_codes import helpers

# this file is used to create the constrain files for F matrix. We could capture the relationship between certain 
# m/z and other m/zs inside one component as the restriction when run the model.

# constrianed all other m/zs by m/z 29 in HOA
file = "/data/user/wu_j1/Zurich-ToF-ACSM/mz115/processed/input_matrices_martavia_noised_20220315_winter.h5"
components = ["A_DEC_W_037_HOA_t"]
outputfile = "/data/user/wu_j1/Zurich-ToF-ACSM/mz115/processed/input_matrices_martavia_noised_20220315_winter_HOA_full_restriction.h5"
res = helpers.construct_restrictions(
    F_file=file,
    G_file=None,
    F_to_restrict=components,
    G_to_restrict=None,
    reference_mz=29,
    reference_time=None,
)
helpers.dump_hdf5(outputfile, **res)

# constrian all other m/zs by m/z 29 in HOA and BBOA
file = "/data/user/wu_j1/Zurich-ToF-ACSM/mz115/processed/input_matrices_martavia_noised_20220315_winter.h5"
components = ["A_DEC_W_037_HOA_t", "A_DEC_W_035_BBOA_t"]
outputfile = "/data/user/wu_j1/Zurich-ToF-ACSM/mz115/processed/input_matrices_martavia_noised_20220315_winter_HOA_BBOA_full_restriction.h5"

res = helpers.construct_restrictions(
    F_file=file,
    G_file=None,
    F_to_restrict=components,
    G_to_restrict=None,
    reference_mz=29,
    reference_time=None,
)
helpers.dump_hdf5(outputfile, **res)

# constrian all other m/zs by m/z 29 in HOA, BBOA and SOAbb
file = "/data/user/wu_j1/Zurich-ToF-ACSM/mz115/processed/input_matrices_martavia_noised_20220315_winter.h5"
components = ["A_DEC_W_037_HOA_t", "A_DEC_W_035_BBOA_t", "SOAbb_t"]
outputfile = "/data/user/wu_j1/Zurich-ToF-ACSM/mz115/processed/input_matrices_martavia_noised_20220315_winter_HOA_BBOA_SOAbb_partial_restriction.h5"

res = helpers.construct_restrictions(
    F_file=file,
    G_file=None,
    F_to_restrict=components,
    G_to_restrict=None,
    reference_mz=29,
    reference_time=None,
)
helpers.dump_hdf5(outputfile, **res)

# constrian m/z 45,57,60 by m/z 43 in HOA and BBOA
file = "/data/user/wu_j1/Zurich-ToF-ACSM/mz115/processed/input_matrices_martavia_noised_20220315_winter.h5"
components = ["A_DEC_W_037_HOA_t", "A_DEC_W_035_BBOA_t"]
outputfile = "/data/user/wu_j1/Zurich-ToF-ACSM/mz115/processed/input_matrices_martavia_noised_20220315_winter_HOA_BBOA_partial_restriction.h5"

res = helpers.construct_restrictions(
    F_file=file,
    G_file=None,
    F_to_restrict=components,
    G_to_restrict=None,
    reference_mz=43,
    restricted_mz=[45, 57, 60],
    reference_time=None,
    restriction_type="partial",
)
helpers.dump_hdf5(outputfile, **res)

# constrian m/z 45,57,60 by m/z 43 in HOA 
file = "/data/user/wu_j1/Zurich-ToF-ACSM/mz115/processed/input_matrices_martavia_noised_20220315_winter.h5"
components = ["A_DEC_W_037_HOA_t"]
outputfile = "/data/user/wu_j1/Zurich-ToF-ACSM/mz115/processed/input_matrices_martavia_noised_20220315_winter_HOA_partial_restriction.h5"

res = helpers.construct_restrictions(
    F_file=file,
    G_file=None,
    F_to_restrict=components,
    G_to_restrict=None,
    reference_mz=43,
    restricted_mz=[45, 57, 60],
    reference_time=None,
    restriction_type="partial",
)
helpers.dump_hdf5(outputfile, **res)
