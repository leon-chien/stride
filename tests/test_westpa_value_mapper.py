from __future__ import annotations

import numpy as np

from stride.westpa_plugin import StrideValueBinMapper, ValueMapperConfig


def test_stride_value_bin_mapper_assigns_westpa_style_output() -> None:
    mapper = StrideValueBinMapper(
        ValueMapperConfig(
            num_bins=4,
            score_coord_dim=1,
            min_score=0.0,
            max_score=1.0,
        )
    )

    coords = np.asarray(
        [
            [0.0, 0.05],
            [0.0, 0.30],
            [0.0, 0.70],
            [0.0, 1.20],
        ],
        dtype=np.float32,
    )

    assignments = mapper.assign(coords)

    assert mapper.nbins == 4
    assert assignments.dtype == np.uint16
    assert assignments.tolist() == [0, 1, 2, 3]


def test_stride_value_bin_mapper_respects_mask_and_output() -> None:
    mapper = StrideValueBinMapper(ValueMapperConfig(num_bins=2))

    coords = np.asarray([[0.1], [0.9], [0.2]], dtype=np.float32)
    output = np.asarray([9, 9, 9], dtype=np.uint16)
    mask = np.asarray([True, False, True])

    returned = mapper.assign(coords, mask=mask, output=output)

    assert returned is output
    assert output.tolist() == [0, 9, 0]
