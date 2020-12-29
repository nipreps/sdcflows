"""Test the fieldmap reportlets."""
from pathlib import Path
import pytest
from ..reportlets import FieldmapReportlet
from ...utils.epimanip import epi_mask


@pytest.mark.parametrize("mask", [True, False])
@pytest.mark.parametrize("apply_mask", [True, False])
def test_FieldmapReportlet(tmpdir, outdir, testdata_dir, mask, apply_mask):
    """Generate one reportlet."""
    tmpdir.chdir()

    if not outdir:
        outdir = Path.cwd()

    report = FieldmapReportlet(
        reference=str(testdata_dir / "epi.nii.gz"),
        moving=str(testdata_dir / "epi.nii.gz"),
        fieldmap=str(testdata_dir / "topup-field.nii.gz"),
        out_report=str(
            outdir / f"test-fieldmap{'-masked' * mask}{'-zeroed' * apply_mask}.svg"
        ),
        apply_mask=apply_mask,
    )

    if mask:
        report.inputs.mask = epi_mask(str(testdata_dir / "epi.nii.gz"))

    report.run()
