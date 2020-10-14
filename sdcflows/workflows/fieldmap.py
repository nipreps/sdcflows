DEFAULT_MEMORY_MIN_GB = 0.01


def init_fmap_preproc_wf(
    *,
    bids_root,
    input_data,
    output_dir,
    debug=False,
    omp_nthreads=None,
    name='fmap_preproc_wf',
):
    """
    Stage the fieldmap preprocessing steps of *SDCFlows*.

    This workflow takes at the input a set of images from which the fieldmap
    could be theoretically estimated by one of the available implementations.
    The workflow API is thought out to be consistent with that of
    :py:func:`~smriprep.workflows.anatomical.init_anat_preproc_wf` in the sense that
    it takes some specific inputs and generates some expected derivatives.

    The particular estimation strategy is inferred from the actual inputs
    given by ``input_data``.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes
            from sdcflows.workflows.fieldmap import init_fmap_preproc_wf
            wf = init_fmap_preproc_wf(
                bids_root='.',
                input_data=[
                    {
                        "filename": "/data/sub-01/fmap/sub-01_epi.nii.gz",
                        "PhaseEncodingDirection": "j",
                        "TotalReadoutTime": "0.065",
                        "IntendedFor": "func/sub-01_task-rest_bold.nii.gz"
                    },
                    {
                        "filename": "/data/sub-01/func/sub-01_task-rest_sbref.nii.gz",
                        "PhaseEncodingDirection": "j-",
                        "TotalReadoutTime": "0.065"
                    }
                ],
                output_dir="/data/derivatives/sdcflows-1.3.1",
            )

    Parameters
    ----------
    bids_root : :obj:`str`
        Path of the input BIDS dataset root
    output_dir : :obj:`str`
        Directory in which to save derivatives
    input_data : :obj:`list` of :obj:`dict`
        The input data that can be used to estimate a fieldmap.
    debug : :obj:`bool`
        Enable debugging outputs
    omp_nthreads : :obj:`int`
        Maximum number of threads an individual process may use
    name : :obj:`str`, optional
        Workflow name (default: anat_preproc_wf)

    """
    return None
