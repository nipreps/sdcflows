## THIS FILE CONTAINS LEGACY CODE JUST IN CASE WE WANT TO LOOK UP SOMETHING FROM
## base.py IN THE FUTURE. SHOULD BE DELETED ASAP.

    # inputnode = pe.Node(niu.IdentityInterface(
    #     fields=['fieldmaps', 't1', 'func', 'sbref']), name='inputnode')

    # for key in subject_list.keys():
    #     if key != 'func':
    #         setattr(inputnode.inputs, key, subject_list[key])

    # inputfmri = pe.Node(niu.IdentityInterface(
    #                     fields=['func']), name='inputfmri')
    # inputfmri.iterables = [('func', subject_list['func'])]

    # outputnode = pe.Node(niu.IdentityInterface(
    #     fields=['fieldmap', 'corrected_sbref', 'fmap_mag', 'fmap_mag_brain',
    #             't1', 'stripped_epi', 'corrected_epi_mean', 'sbref_brain',
    #             'stripped_epi_mask', 'stripped_t1', 't1_segmentation',
    #             't1_2_mni', 't1_wm_seg']),
    #     name='outputnode'
    # )

    # try:
    #     fmap_wf = fieldmap_decider(getattr(inputnode.inputs, 'fieldmaps'), settings)
    # except NotImplementedError:
    #     fmap_wf = None

    # # Reorient EPI to RAS
    # split = pe.Node(fsl.Split(dimension='t'), name='SplitEPI')
    # orient = pe.MapNode(fs.MRIConvert(out_type='niigz', out_orientation='RAS'),
    #                     iterfield=['in_file'], name='ReorientEPI')
    # merge = pe.Node(fsl.Merge(dimension='t'), name='MergeEPI')

    #
    # epi_hmc_wf = epi_hmc(settings=settings, sbref_present=sbref_present)
    # epi_mni_trans_wf = epi_mni_transformation(settings=settings)

    # #  Connecting Workflow pe.Nodes
    # workflow.connect([
    #     (inputfmri, split, [('func', 'in_file')]),
    #     (split, orient, [('out_files', 'in_file')]),
    #     (orient, merge, [('out_file', 'in_files')]),

    #     (inputnode, t1w_preproc, [('t1', 'inputnode.t1')]),
    #     (merge, epi_hmc_wf, [('merged_file', 'inputnode.epi_ras')]),
    #     (merge, epi_mni_trans_wf, [('merged_file', 'inputnode.epi_ras')]),

    #     # These are necessary sources for the DerivativesDataSink
    #     (inputfmri, epi_hmc_wf, [('func', 'inputnode.epi')]),
    #     (inputfmri, epi_mni_trans_wf, [('func', 'inputnode.epi')])
    # ])

    # if not sbref_present:
    #     epi_2_t1 = epi_mean_t1_registration(settings=settings)
    #     workflow.connect([
    #         (inputfmri, epi_2_t1, [('func', 'inputnode.epi')]),
    #         (epi_hmc_wf, epi_2_t1, [('outputnode.epi_mean', 'inputnode.epi_mean')]),
    #         (t1w_preproc, epi_2_t1, [('outputnode.t1_brain', 'inputnode.t1_brain'),
    #                                  ('outputnode.t1_seg', 'inputnode.t1_seg')]),
    #         (epi_2_t1, epi_mni_trans_wf, [('outputnode.mat_epi_to_t1', 'inputnode.mat_epi_to_t1')]),

    #         (epi_hmc_wf, epi_mni_trans_wf, [('outputnode.xforms', 'inputnode.hmc_xforms'),
    #                                         ('outputnode.epi_mask', 'inputnode.epi_mask')]),
    #         (t1w_preproc, epi_mni_trans_wf, [('outputnode.t1_brain', 'inputnode.t1'),
    #                                          ('outputnode.t1_2_mni_forward_transform',
    #                                           'inputnode.t1_2_mni_forward_transform')])
    #     ])

    # if fmap_wf:
    #     sbref_wf = sbref_workflow(settings=settings)
    #     sbref_wf.inputs.inputnode.hmc_mats = []  # FIXME: plug MCFLIRT output here
    #     sbref_t1 = sbref.sbref_t1_registration(settings=settings)
    #     unwarp_wf = epi_unwarp(settings=settings)
    #     sepair_wf = se_pair_workflow(settings=settings)
    #     workflow.connect([
    #         (inputnode, sepair_wf, [('fieldmaps', 'inputnode.input_images')]),
    #         (inputnode, sbref_wf, [('sbref', 'inputnode.sbref')]),
    #         (inputfmri, unwarp_wf, [('func', 'inputnode.epi')]),
    #         (sepair_wf, sbref_wf, [
    #             ('outputnode.mag_brain', 'inputnode.fmap_ref_brain'),
    #             ('outputnode.fmap_mask', 'inputnode.fmap_mask'),
    #             ('outputnode.fieldmap', 'inputnode.fieldmap')
    #         ]),
    #         (sbref_wf, sbref_t1, [
    #             ('outputnode.sbref_unwarped', 'inputnode.sbref_brain')]),
    #         (t1w_preproc, sbref_t1, [
    #             ('outputnode.t1_brain', 'inputnode.t1_brain'),
    #             ('outputnode.t1_seg', 'inputnode.t1_seg')]),
    #         (sbref_wf, unwarp_wf, [
    #             ('outputnode.sbref_unwarped', 'inputnode.sbref_brain')]),
    #         (sbref_wf, epi_hmc_wf, [
    #             ('outputnode.sbref_unwarped', 'inputnode.sbref_brain')]),
    #         (epi_hmc_wf, unwarp_wf, [
    #             ('outputnode.epi_brain', 'inputnode.epi_brain')]),
    #         (sepair_wf, unwarp_wf, [
    #             ('outputnode.fmap_mask', 'inputnode.fmap_mask'),
    #             ('outputnode.fieldmap', 'inputnode.fieldmap')
    #         ]),
    #     ])
