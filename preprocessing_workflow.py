# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 17:35:40 2015

@author: craigmoodie
"""

################################################### Setting up Workflow ###########################################

from nipype.interfaces.utility import Function
from nipype.pipeline.engine import Workflow, Node, MapNode
from variables_preprocessing import data_dir, work_dir, subject_list, plugin, plugin_args



Preprocessing_workflow = Workflow(name="Final_preprocessing_workflow")
Preprocessing_workflow.base_dir = work_dir


from nipype import SelectFiles
templates = dict(fieldmaps="sub-{subject_id}/fmap/sub-{subject_id}*_epi.nii.gz",
                 fieldmaps_meta="sub-{subject_id}/fmap/sub-{subject_id}*_epi.json",
                 epi="sub-{subject_id}/func/sub-{subject_id}_task-rest_acq-LR_run-1_bold.nii.gz",
                 epi_meta="sub-{subject_id}/func/sub-{subject_id}_task-rest_acq-LR_run-1_bold.nii.gz",
                 sbref="sub-{subject_id}/func/sub-{subject_id}_task-rest_acq-LR_run-1_sbref.nii.gz",
                 sbref_meta="sub-{subject_id}/func/sub-{subject_id}_task-rest_acq-LR_run-1_sbref.json",
                 t1="sub-{subject_id}/anat/sub-{subject_id}_run-1_T1w.nii.gz")
file_list = Node(SelectFiles(templates), name = "File_Selection")
file_list.inputs.base_directory = data_dir
file_list.iterables = [("subject_id", subject_list)]





############################################################# Prep Steps ############################################



#### Merge the SE scouts

from nipype.interfaces.fsl import Merge
fslmerge = Node(Merge(), name="Merge_Fieldmaps")
fslmerge.inputs.dimension = 't'
fslmerge.inputs.output_type = 'NIFTI_GZ'



#### Run MCFLIRT to put the merged SE scouts into to roughly the same space in order to better estimate the field with SBRef as a target here

from nipype.interfaces.fsl import MCFLIRT
motion_correct_SE_maps = Node(MCFLIRT(), name="Motion_Correction")
motion_correct_SE_maps.inputs.output_type = "NIFTI_GZ"



###### Skull strip EPI  (try ComputeMask(BaseInterface))

from nipype.interfaces.fsl import BET
EPI_BET = Node(BET(), name = "EPI_BET")
EPI_BET.inputs.mask = True
EPI_BET.inputs.functional = True
EPI_BET.inputs.frac = 0.6




##### Skull strip SBRef to get reference brain

SBRef_BET = Node(BET(), name = "SBRef_BET")
SBRef_BET.inputs.mask = True
SBRef_BET.inputs.frac = 0.6
SBRef_BET.inputs.robust = True


#### Skull strip the SBRef with ANTS Brain Extraction

#from nipype.interfaces.ants.segmentation import BrainExtraction
#SBRef_skull_strip = Node(BrainExtraction(), name = "Ants_T1_Brain_Extraction")
#SBRef_skull_strip.inputs.dimension = 3
#SBRef_skull_strip.inputs.brain_template = "/home/cmoodie/Oasis_MICCAI2012-Multi-Atlas-Challenge-Data/T_template0.nii.gz"
#SBRef_skull_strip.inputs.brain_probability_mask = "/home/cmoodie/Oasis_MICCAI2012-Multi-Atlas-Challenge-Data/T_template0_BrainCerebellumProbabilityMask.nii.gz"
#SBRef_skull_strip.inputs.extraction_registration_mask = "/home/cmoodie/Oasis_MICCAI2012-Multi-Atlas-Challenge-Data/T_template0_BrainCerebellumRegistrationMask.nii.gz"



#### T1 Bias Field Correction 

from nipype.interfaces.ants import N4BiasFieldCorrection
n4 = Node(N4BiasFieldCorrection(), name = "Bias_Field_Correction")
n4.inputs.dimension = 3
n4.inputs.output_image = "Bias_Corrected_T1.nii.gz"
n4.inputs.bspline_fitting_distance = 300
n4.inputs.shrink_factor = 3



#### Skull strip the T1 with ANTS Brain Extraction

from nipype.interfaces.ants.segmentation import BrainExtraction
T1_skull_strip = Node(BrainExtraction(), name = "Ants_T1_Brain_Extraction")
T1_skull_strip.inputs.dimension = 3
T1_skull_strip.inputs.brain_template = "/home/cmoodie/Oasis_MICCAI2012-Multi-Atlas-Challenge-Data/T_template0.nii.gz"
T1_skull_strip.inputs.brain_probability_mask = "/home/cmoodie/Oasis_MICCAI2012-Multi-Atlas-Challenge-Data/T_template0_BrainCerebellumProbabilityMask.nii.gz"
T1_skull_strip.inputs.extraction_registration_mask = "/home/cmoodie/Oasis_MICCAI2012-Multi-Atlas-Challenge-Data/T_template0_BrainCerebellumRegistrationMask.nii.gz"



#fast -o fast_test -N -v ../Preprocessing_test_workflow/_subject_id_S2529LVY1263171/Bias_Field_Correction/sub-S2529LVY1263171_run-1_T1w_corrected.nii.gz

from nipype.interfaces.fsl import FAST
T1_seg = Node(FAST(), name = "T1_Segmentation")
T1_seg.inputs.no_bias = True
T1_seg.inputs.output_type = 'NIFTI_GZ'
T1_seg.inputs.verbose = True



#### Affine transform of T1 segmentation into SBRref space

from nipype.interfaces.fsl import FLIRT
flt_wmseg_sbref = Node(FLIRT(bins=640, cost_func='mutualinfo'), name = "WMSeg_2_SBRef_Brain_Affine_Transform")
flt_wmseg_sbref.inputs.output_type = "NIFTI_GZ"
flt_wmseg_sbref.inputs.out_matrix_file ="wmseg_to_sbref_brain.mat"
flt_wmseg_sbref.inputs.out_file = "wmseg_flirt_sbref_brain.nii.gz"
flt_wmseg_sbref.inputs.dof = 6

############################################################# Topup Steps #################################################


#### Create the acquisition parameters file that topup needs

def create_encoding_file(fieldmaps, fieldmaps_meta):
    import json, nibabel, os
    with open("parameters.txt", "w") as parameters_file:
        for fieldmap, fieldmap_meta in zip(fieldmaps, fieldmaps_meta):
            meta = json.load(open(fieldmap_meta))
            pedir = {'x': 0, 'y': 1, 'z': 2}
            line_values = [0, 0, 0, meta["TotalReadoutTime"]]
            line_values[pedir[meta["PhaseEncodingDirection"][0]]] = 1 + (-2*(len(meta["PhaseEncodingDirection"]) == 2))
            for i in range (nibabel.load(fieldmap).shape[-1]):
                parameters_file.write(" ".join([str(i) for i in line_values]) + "\n")
    return os.path.abspath("parameters.txt")
    
create_parameters_node = Node(Function(input_names=["fieldmaps", "fieldmaps_meta"], 
                                       output_names=["parameters_file"],
                                       function=create_encoding_file), 
                                       name="Create_Parameters",updatehash = True)



#### Run topup to estimate filed distortions

from nipype.interfaces.fsl import TOPUP
topup = Node(TOPUP(), name="TopUp")
topup.inputs.output_type = "NIFTI_GZ"
topup.inputs.out_field = "Topup_Fieldmap_rad.nii.gz"
topup.inputs.out_corrected = "fieldmap_mag_topup_corrected.nii.gz"





################################# Distortion Corretion using the TopUp Fieldmap ##############################################


############ Convert topup fieldmap to rad/s [ 1 Hz = 6.283 rad/s]

from nipype.interfaces.fsl import BinaryMaths
fmap_scale = Node(BinaryMaths(), name = "Scale_Fieldmap")
fmap_scale.inputs.operand_value = 6.283
fmap_scale.inputs.operation = 'mul'
fmap_scale.inputs.output_type = 'NIFTI_GZ'



###### Skull strip SE Fieldmap magnitude image to get reference brain and mask

fmap_mag_BET = Node(BET(), name = "Fmap_Mag_BET")
fmap_mag_BET.inputs.mask = True
fmap_mag_BET.inputs.robust = True
#### Might want to turn off bias reduction if it is being done in a separate node!
#fmap_mag_BET.inputs.reduce_bias = True




#### Unwarp SBRef using Fugue  (N.B. duplicated in epi_reg_workflow!!!!!)

from nipype.interfaces.fsl.preprocess import FUGUE
fugue_sbref = Node(FUGUE(), name = "SBRef_Unwarping")
fugue_sbref.inputs.unwarp_direction = 'x'
fugue_sbref.inputs.dwell_time = 0.000700012460221792
fugue_sbref.inputs.output_type = "NIFTI_GZ"



strip_corrected_sbref = Node(BET(), name = "BET_Corrected_SBRef")
strip_corrected_sbref.inputs.mask = True
strip_corrected_sbref.inputs.frac = 0.6
strip_corrected_sbref.inputs.robust = True



##### Run MCFLIRT to get motion matrices

#### Motion Correction of the EPI with SBRef as Target

motion_correct_epi = Node(MCFLIRT(), name="Motion_Correction_EPI")
motion_correct_epi.inputs.output_type = "NIFTI_GZ"
motion_correct_epi.inputs.save_mats = True    



################################ Run the commands from epi_reg_dof ###########################################################


################## do a standard flirt pre-alignment

#$FSLDIR/bin/flirt -ref ${vrefbrain} -in ${vepi} -dof ${dof} -omat ${vout}_init.mat

flt_epi_sbref = Node(FLIRT(bins=640, cost_func='mutualinfo'), name = "EPI_2_SBRef_Brain_Affine_Transform")
flt_epi_sbref.inputs.output_type = "NIFTI_GZ"
flt_epi_sbref.inputs.out_matrix_file ="epi_to_sbref_brain.mat"
flt_epi_sbref.inputs.out_file = "epi_2_sbref_brain.nii.gz"
flt_epi_sbref.inputs.dof = 6

################## WITH FIELDMAP (unwarping steps)

#$FSLDIR/bin/flirt -in ${fmapmagbrain} -ref ${vrefbrain} -dof ${dof} -omat ${vout}_fieldmap2str_init.mat

flt_fmap_mag_brain_sbref_brain = Node(FLIRT(bins=640, cost_func='mutualinfo'), name = "Fmap_Mag_Brain_2_SBRef_Brain_Affine_Transform")
flt_fmap_mag_brain_sbref_brain.inputs.output_type = "NIFTI_GZ"
flt_fmap_mag_brain_sbref_brain.inputs.out_matrix_file = "fmap_mag_brain_to_sbref_brain.mat"
flt_fmap_mag_brain_sbref_brain.inputs.out_file = "fmap_mag_brain_flirt_sbref_brain.nii.gz"
flt_fmap_mag_brain_sbref_brain.inputs.dof = 6


#$FSLDIR/bin/flirt -in ${fmapmaghead} -ref ${vrefhead} -dof ${dof} -init ${vout}_fieldmap2str_init.mat -omat ${vout}_fieldmap2str.mat -out ${vout}_fieldmap2str -nosearch

flt_fmap_mag_sbref = Node(FLIRT(bins=640, cost_func='mutualinfo'), name = "Fmap_Mag_2_SBRef_Affine_Transform")
flt_fmap_mag_sbref.inputs.output_type = "NIFTI_GZ"
flt_fmap_mag_sbref.inputs.out_matrix_file = "fmap_mag_to_sbref.mat"
flt_fmap_mag_sbref.inputs.out_file = "fmap_mag_flirt_sbref.nii.gz"
flt_fmap_mag_sbref.inputs.dof = 6
flt_fmap_mag_sbref.inputs.no_search = True



################## unmask the fieldmap (necessary to avoid edge effects)

#$FSLDIR/bin/fslmaths ${fmapmagbrain} -abs -bin ${vout}_fieldmaprads_mask

from nipype.interfaces.fsl import UnaryMaths
fmapmagbrain_abs = Node(UnaryMaths(), name = "Abs_Fieldmap_Mag_Brain")
fmapmagbrain_abs.inputs.operation = 'abs'
fmapmagbrain_abs.inputs.output_type = 'NIFTI_GZ'

from nipype.interfaces.fsl import BinaryMaths
fmapmagbrain_bin = Node(UnaryMaths(), name = "Binarize_Fieldmap_Mag_Brain")
fmapmagbrain_bin.inputs.operation = 'bin'
fmapmagbrain_bin.inputs.output_type = 'NIFTI_GZ'



#$FSLDIR/bin/fslmaths ${fmaprads} -abs -bin -mul ${vout}_fieldmaprads_mask ${vout}_fieldmaprads_mask

fmap_abs = Node(UnaryMaths(), name = "Abs_Fieldmap")
fmap_abs.inputs.operation = 'abs'
fmap_abs.inputs.output_type = 'NIFTI_GZ'

fmap_bin = Node(UnaryMaths(), name = "Binarize_Fieldmap")
fmap_bin.inputs.operation = 'bin'
fmap_bin.inputs.output_type = 'NIFTI_GZ'

fmap_mul = Node(BinaryMaths(), name = "Fmap_Multiplied_by_Mask")
fmap_mul.inputs.operation = 'mul'
fmap_mul.inputs.output_type = 'NIFTI_GZ'



#$FSLDIR/bin/fugue --loadfmap=${fmaprads} --mask=${vout}_fieldmaprads_mask --unmaskfmap --savefmap=${vout}_fieldmaprads_unmasked --unwarpdir=${fdir}   # the direction here should take into account the initial affine (it needs to be the direction in the EPI)

fugue_unmask = Node(FUGUE(), name = "Fmap_Unmasking")
fugue_unmask.inputs.unwarp_direction = 'x'
fugue_unmask.inputs.dwell_time = 0.000700012460221792
fugue_unmask.inputs.save_unmasked_fmap = True
fugue_unmask.inputs.output_type = "NIFTI_GZ"



################## the following is a NEW HACK to fix extrapolation when fieldmap is too small

#$FSLDIR/bin/applywarp -i ${vout}_fieldmaprads_unmasked -r ${vrefhead} --premat=${vout}_fieldmap2str.mat -o ${vout}_fieldmaprads2str_pad0

from nipype.interfaces.fsl import ApplyWarp
aw_fmap_unmasked_sbref = Node(ApplyWarp(), name="Apply_Warp_Fmap_Unmasked_2_SBRef")
aw_fmap_unmasked_sbref.inputs.relwarp = True
aw_fmap_unmasked_sbref.inputs.output_type = 'NIFTI_GZ'



#$FSLDIR/bin/fslmaths ${vout}_fieldmaprads2str_pad0 -abs -bin ${vout}_fieldmaprads2str_innermask

fmap_unmasked_abs = Node(UnaryMaths(), name = "Abs_Fmap_Unmasked_Warp")
fmap_unmasked_abs.inputs.operation = 'abs'
fmap_unmasked_abs.inputs.output_type = 'NIFTI_GZ'

fmap_unmasked_bin = Node(UnaryMaths(), name = "Binarize_Fmap_Unmasked_Warp")
fmap_unmasked_bin.inputs.operation = 'bin'
fmap_unmasked_bin.inputs.output_type = 'NIFTI_GZ'



#$FSLDIR/bin/fugue --loadfmap=${vout}_fieldmaprads2str_pad0 --mask=${vout}_fieldmaprads2str_innermask --unmaskfmap --unwarpdir=${fdir} --savefmap=${vout}_fieldmaprads2str_dilated

fugue_dilate = Node(FUGUE(), name = "Fmap_Dilating")
fugue_dilate.inputs.unwarp_direction = 'x'
fugue_dilate.inputs.dwell_time = 0.000700012460221792
fugue_dilate.inputs.save_unmasked_fmap = True
fugue_dilate.inputs.output_type = "NIFTI_GZ"



#$FSLDIR/bin/fslmaths ${vout}_fieldmaprads2str_dilated ${vout}_fieldmaprads2str   
# !!!! Don't need to do this since this just does the same thing as a "mv" command. Just connect previous fugue node directly to subsequent flirt command.



################## run bbr to SBRef target with fieldmap and T1 seg in SBRef space


#$FSLDIR/bin/flirt -ref ${vrefhead} -in ${vepi} -dof ${dof} -cost bbr -wmseg ${vout}_fast_wmseg -init ${vout}_init.mat -omat ${vout}.mat -out ${vout}_1vol -schedule ${FSLDIR}/etc/flirtsch/bbr.sch -echospacing ${dwell} -pedir ${pe_dir} -fieldmap ${vout}_fieldmaprads2str $wopt

flt_bbr = Node(FLIRT(bins=640, cost_func='bbr'), name = "Flirt_BBR")
flt_bbr.inputs.output_type = "NIFTI_GZ"
flt_bbr.inputs.out_matrix_file = "flirt_bbr.mat"
flt_bbr.inputs.out_file = "flirt_bbr_w_fmap.nii.gz"
flt_bbr.inputs.dof = 6
flt_bbr.inputs.schedule = "/share/sw/free/fsl/5.0.7/fsl/etc/flirtsch/bbr.sch"
flt_bbr.inputs.echospacing = 0.000700012460221792
flt_bbr.inputs.pedir = 1
 


################## make equivalent warp fields 
 
#$FSLDIR/bin/convert_xfm -omat ${vout}_inv.mat -inverse ${vout}.mat

from nipype.interfaces.fsl import ConvertXFM
invt_bbr = Node(ConvertXFM(), name= "BBR_Inverse_Transform")
invt_bbr.inputs.invert_xfm = True
invt_bbr.inputs.out_file = 'BBR_inverse.mat'



#$FSLDIR/bin/convert_xfm -omat ${vout}_fieldmaprads2epi.mat -concat ${vout}_inv.mat ${vout}_fieldmap2str.mat

concat_mats = Node(ConvertXFM(), name= "BBR_Concat")
concat_mats.inputs.concat_xfm = True




#$FSLDIR/bin/applywarp -i ${vout}_fieldmaprads_unmasked -r ${vepi} --premat=${vout}_fieldmaprads2epi.mat -o ${vout}_fieldmaprads2epi

aw_fmap_unmasked_epi = Node(ApplyWarp(), name="Apply_Warp_Fmap_Unmasked_2_EPI")
aw_fmap_unmasked_epi.inputs.relwarp = True
aw_fmap_unmasked_epi.inputs.output_type = 'NIFTI_GZ'



#$FSLDIR/bin/fslmaths ${vout}_fieldmaprads2epi -abs -bin ${vout}_fieldmaprads2epi_mask

fieldmaprads2epi_abs = Node(UnaryMaths(), name = "Abs_Fmap_2_EPI_Unmasked_Warp")
fieldmaprads2epi_abs.inputs.operation = 'abs'
fieldmaprads2epi_abs.inputs.output_type = 'NIFTI_GZ'

fieldmaprads2epi_bin = Node(UnaryMaths(), name = "Binarize_Fmap_2_EPI_Unmasked_Warp")
fieldmaprads2epi_bin.inputs.operation = 'bin'
fieldmaprads2epi_bin.inputs.output_type = 'NIFTI_GZ'



#$FSLDIR/bin/fugue --loadfmap=${vout}_fieldmaprads2epi --mask=${vout}_fieldmaprads2epi_mask --saveshift=${vout}_fieldmaprads2epi_shift --unmaskshift --dwell=${dwell} --unwarpdir=${fdir}
   
fugue_shift = Node(FUGUE(), name = "Fmap_Shift")
fugue_shift.inputs.unwarp_direction = 'x'
fugue_shift.inputs.dwell_time = 0.000700012460221792
fugue_shift.inputs.save_unmasked_shift = True
fugue_shift.inputs.output_type = "NIFTI_GZ"
   
   
#$FSLDIR/bin/convertwarp -r ${vrefhead} -s ${vout}_fieldmaprads2epi_shift --postmat=${vout}.mat -o ${vout}_warp --shiftdir=${fdir} --relout
   
from nipype.interfaces.fsl import ConvertWarp
convert_fmap_shift = MapNode(ConvertWarp(), name = "Convert_Fieldmap_Shift", iterfield=["premat"])
convert_fmap_shift.inputs.out_relwarp = True
convert_fmap_shift.inputs.shift_direction = 'x'



from nipype.interfaces.fsl import Split
split_epi = Node(Split(), "Split_EPI")
split_epi.inputs.output_type = "NIFTI_GZ"
split_epi.inputs.dimension = 't'


#$FSLDIR/bin/applywarp -i ${vepi} -r ${vrefhead} -o ${vout} -w ${vout}_warp --interp=spline --rel

aw_final = MapNode(ApplyWarp(), name="Apply_Final_Warp", iterfield=["field_file", "in_file"])
aw_final.inputs.relwarp = True
aw_final.inputs.output_type = 'NIFTI_GZ'



merge_epi = Node(Merge(), "Merge_EPI")
merge_epi.inputs.output_type = "NIFTI_GZ"
merge_epi.inputs.dimension = 't'



############################################ BBR of Unwarped and SBRef-Registered EPI to T1 ###################################


from nipype.interfaces.fsl import MeanImage
epi_mean = Node(MeanImage(), name = "EPI_mean_volume")
epi_mean.inputs.output_type = "NIFTI_GZ"
epi_mean.inputs.dimension = 'T'


#epi_reg --epi=sub-S2529LVY1263171_task-nback_run-1_bold_brain --t1=../Preprocessing_test_workflow/_subject_id_S2529LVY1263171/Bias_Field_Correction/sub-S2529LVY1263171_run-1_T1w_corrected.nii.gz --t1brain=sub-S2529LVY1263171_run-1_T1w_corrected_bet_brain.nii.gz --out=sub-S2529LVY1263171/func/sub-S2529LVY1263171_task-nback_run-1_bold_undistorted --fmap=Topup_Fieldmap_rad.nii.gz --fmapmag=fieldmap_topup_corrected.nii.gz --fmapmagbrain=Magnitude_brain.nii.gz --echospacing=0.000700012460221792 --pedir=x- -v

flt_sbref_brain_t1_brain = Node(FLIRT(bins=640, cost_func='mutualinfo'), name = "SBRef_Brain_2_T1_Brain_Affine_Transform")
flt_sbref_brain_t1_brain.inputs.output_type = "NIFTI_GZ"
flt_sbref_brain_t1_brain.inputs.out_matrix_file = "sbref_brain_to_sbref_brain.mat"
flt_sbref_brain_t1_brain.inputs.out_file = "sbref_brain_flirt_sbref_brain.nii.gz"
flt_sbref_brain_t1_brain.inputs.dof = 6

flt_sbref_2_T1 = Node(FLIRT(bins=640, cost_func='mutualinfo'), name = "SBRef_2_T1_Affine_Transform")
flt_sbref_2_T1.inputs.output_type = "NIFTI_GZ"
flt_sbref_2_T1.inputs.out_matrix_file ="sbref_to_T1_brain.mat"
flt_sbref_2_T1.inputs.out_file = "sbref_2_T1.nii.gz"
flt_sbref_2_T1.inputs.dof = 6


bbr_sbref_2_T1 = Node(FLIRT(bins=640, cost_func='bbr'), name = "BBR_SBRef_to_T1")
bbr_sbref_2_T1.inputs.output_type = "NIFTI_GZ"
bbr_sbref_2_T1.inputs.out_matrix_file = "bbr.mat"
bbr_sbref_2_T1.inputs.out_file = "bbr_sbref_2_T1.nii.gz"
bbr_sbref_2_T1.inputs.dof = 6
bbr_sbref_2_T1.inputs.schedule = "/share/sw/free/fsl/5.0.7/fsl/etc/flirtsch/bbr.sch"
bbr_sbref_2_T1.inputs.echospacing = 0.000700012460221792
bbr_sbref_2_T1.inputs.pedir = 1





######################################################### Ants registration from T1 to MNI ###############################

from nipype.interfaces.ants import Registration
Ants = Node(Registration(), name = "T1_2_MNI_Registration")
Ants.inputs.fixed_image = "/share/sw/free/fsl/5.0.7/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz"
Ants.inputs.metric = ['Mattes'] * 3 + [['Mattes', 'CC']]
Ants.inputs.metric_weight = [1] * 3 + [[0.5, 0.5]]
Ants.inputs.dimension = 3
Ants.inputs.write_composite_transform = True
Ants.inputs.radius_or_number_of_bins = [32] * 3 + [[32, 4]]
Ants.inputs.shrink_factors = [[6, 4, 2]] + [[3, 2, 1]]*2 + [[4, 2, 1]]
Ants.inputs.smoothing_sigmas = [[4, 2, 1]] * 3 + [[1, 0.5, 0]]
Ants.inputs.sigma_units = ['vox'] * 4
Ants.inputs.output_transform_prefix = "ANTS_T1_2_MNI"
Ants.inputs.transforms = ['Translation', 'Rigid', 'Affine', 'SyN']
Ants.inputs.transform_parameters = [(0.1,), (0.1,), (0.1,), (0.2, 3.0, 0.0)]
Ants.inputs.initial_moving_transform_com = True
Ants.inputs.number_of_iterations = ([[10, 10, 10]]*3 + [[1, 5, 3]])
Ants.inputs.convergence_threshold = [1.e-8] * 3 + [-0.01]
Ants.inputs.convergence_window_size = [20] * 3 + [5]
Ants.inputs.sampling_strategy = ['Regular'] * 3 + [[None, None]]
Ants.inputs.sampling_percentage = [0.3] * 3 + [[None, None]]
Ants.inputs.output_warped_image = True
Ants.inputs.use_histogram_matching = [False] * 3 + [True]
Ants.inputs.use_estimate_learning_rate_once = [True] * 4
#Ants.inputs.interpolation = 'NearestNeighbor'




########################################### Transforming parcels from standard to native SBRef space ############################


from nipype.interfaces.ants import ApplyTransforms
at = Node(ApplyTransforms(), name = "Apply_ANTS_transform_MNI_2_T1")
at.inputs.dimension = 3
at.inputs.interpolation = 'NearestNeighbor'
at.inputs.default_value = 0
at.inputs.input_image = "/home/cmoodie/Parcels_MNI_222.nii.gz"

from nipype.interfaces.fsl import ConvertXFM
invt_mat = Node(ConvertXFM(), name= "EpiReg_Inverse_Transform")
invt_mat.inputs.invert_xfm = True
invt_mat.inputs.out_file = 'Epi_reg_inverse.mat'


from nipype.interfaces.fsl import FLIRT
flt_parcels_2_sbref = Node(FLIRT(bins=640, cost_func='mutualinfo'), name = "Parcels_2_EPI_Mean_Affine_w_Inv_Mat")
flt_parcels_2_sbref.inputs.output_type = "NIFTI_GZ"
flt_parcels_2_sbref.inputs.out_file = "parcels_in_native_space.nii.gz"
flt_parcels_2_sbref.inputs.dof = 12
flt_parcels_2_sbref.inputs.interp = 'nearestneighbour'






########################################## Connecting Workflow Nodes ###########################################################

Preprocessing_workflow.connect(file_list, "fieldmaps", fslmerge, "in_files")

Preprocessing_workflow.connect(fslmerge, "merged_file", motion_correct_SE_maps, "in_file")
Preprocessing_workflow.connect(file_list, "sbref", motion_correct_SE_maps, "ref_file")

Preprocessing_workflow.connect(file_list, "epi", EPI_BET, "in_file")

Preprocessing_workflow.connect(file_list, "sbref", SBRef_BET, "in_file")
#Preprocessing_workflow.connect(file_list, "sbref", SBRef_skull_strip, "in_file")

Preprocessing_workflow.connect(file_list, "t1", n4, "input_image")

Preprocessing_workflow.connect(n4, "output_image", T1_skull_strip, "anatomical_image")

Preprocessing_workflow.connect(n4,"output_image", T1_seg, "in_files")

Preprocessing_workflow.connect(T1_seg, "tissue_class_map", flt_wmseg_sbref, "in_file")
Preprocessing_workflow.connect(file_list, "sbref", flt_wmseg_sbref, "reference")

Preprocessing_workflow.connect(file_list, "fieldmaps", create_parameters_node, "fieldmaps")
Preprocessing_workflow.connect(file_list, "fieldmaps_meta", create_parameters_node, "fieldmaps_meta")

Preprocessing_workflow.connect(create_parameters_node, "parameters_file", topup, "encoding_file")
Preprocessing_workflow.connect(motion_correct_SE_maps, "out_file", topup, "in_file")

Preprocessing_workflow.connect(topup, "out_field", fmap_scale, "in_file")

Preprocessing_workflow.connect(topup, "out_corrected", fmap_mag_BET, "in_file")

Preprocessing_workflow.connect(fmap_scale, "out_file", fugue_sbref, "fmap_in_file")
Preprocessing_workflow.connect(fmap_mag_BET, "mask_file", fugue_sbref, "mask_file")
Preprocessing_workflow.connect(file_list, "sbref", fugue_sbref, "in_file")

Preprocessing_workflow.connect(fugue_sbref, "unwarped_file", strip_corrected_sbref, "in_file")

Preprocessing_workflow.connect(EPI_BET, "out_file", motion_correct_epi, "in_file")
Preprocessing_workflow.connect(file_list, "sbref", motion_correct_epi, "ref_file")

Preprocessing_workflow.connect(EPI_BET, "out_file", flt_epi_sbref, "in_file")
Preprocessing_workflow.connect(SBRef_BET, "out_file", flt_epi_sbref, "reference") ### might need to switch to [strip_corrected_sbref, "in_file"] here instead of [SBRef_BET, "out_file"]

Preprocessing_workflow.connect(fmap_mag_BET, "out_file", flt_fmap_mag_brain_sbref_brain, "in_file")
Preprocessing_workflow.connect(SBRef_BET, "out_file", flt_fmap_mag_brain_sbref_brain, "reference") ### might need to switch to [strip_corrected_sbref, "in_file"] here instead of [SBRef_BET, "out_file"]

Preprocessing_workflow.connect(topup, "out_corrected", flt_fmap_mag_sbref, "in_file")
Preprocessing_workflow.connect(fugue_sbref, "unwarped_file", flt_fmap_mag_sbref, "reference")
Preprocessing_workflow.connect(flt_fmap_mag_brain_sbref_brain, "out_matrix_file", flt_fmap_mag_sbref, "in_matrix_file")

Preprocessing_workflow.connect(topup, "out_corrected", fmapmagbrain_abs, "in_file")

Preprocessing_workflow.connect(fmapmagbrain_abs, "out_file", fmapmagbrain_bin, "in_file")

Preprocessing_workflow.connect(fmap_scale, "out_file", fmap_abs, "in_file")

Preprocessing_workflow.connect(fmap_abs, "out_file", fmap_bin, "in_file")

Preprocessing_workflow.connect(fmap_bin, "out_file", fmap_mul, "in_file")
Preprocessing_workflow.connect(fmapmagbrain_bin, "out_file", fmap_mul, "operand_file")

Preprocessing_workflow.connect(fmap_scale, "out_file", fugue_unmask, "fmap_in_file")
Preprocessing_workflow.connect(fmap_mul, "out_file", fugue_unmask, "mask_file")

Preprocessing_workflow.connect(fugue_unmask, "fmap_out_file", aw_fmap_unmasked_sbref, "in_file")
Preprocessing_workflow.connect(flt_fmap_mag_sbref, "out_matrix_file", aw_fmap_unmasked_sbref, "premat")
Preprocessing_workflow.connect(fugue_sbref, "unwarped_file", aw_fmap_unmasked_sbref, "ref_file")

Preprocessing_workflow.connect(aw_fmap_unmasked_sbref, "out_file", fmap_unmasked_abs, "in_file")

Preprocessing_workflow.connect(fmap_unmasked_abs, "out_file", fmap_unmasked_bin, "in_file")

Preprocessing_workflow.connect(aw_fmap_unmasked_sbref, "out_file", fugue_dilate, "fmap_in_file")
Preprocessing_workflow.connect(fmap_unmasked_bin, "out_file", fugue_dilate, "mask_file")

Preprocessing_workflow.connect(EPI_BET, "out_file", flt_bbr, "in_file")
Preprocessing_workflow.connect(fugue_sbref, "unwarped_file", flt_bbr, "reference")
Preprocessing_workflow.connect(fugue_dilate, "fmap_out_file", flt_bbr, "fieldmap") 
Preprocessing_workflow.connect(flt_epi_sbref, "out_matrix_file", flt_bbr, "in_matrix_file") 
Preprocessing_workflow.connect(flt_wmseg_sbref, "out_file", flt_bbr, "wm_seg")
                         
Preprocessing_workflow.connect(flt_bbr, "out_matrix_file", invt_bbr, "in_file")

Preprocessing_workflow.connect(flt_fmap_mag_sbref, "out_matrix_file", concat_mats, "in_file")
Preprocessing_workflow.connect(invt_bbr, "out_file", concat_mats, "in_file2")

Preprocessing_workflow.connect(fugue_unmask, "fmap_out_file", aw_fmap_unmasked_epi, "in_file")
Preprocessing_workflow.connect(EPI_BET, "out_file", aw_fmap_unmasked_epi, "ref_file")
Preprocessing_workflow.connect(concat_mats, "out_file", aw_fmap_unmasked_epi, "premat")

Preprocessing_workflow.connect(aw_fmap_unmasked_epi, "out_file", fieldmaprads2epi_abs, "in_file")

Preprocessing_workflow.connect(fieldmaprads2epi_abs, "out_file", fieldmaprads2epi_bin, "in_file")

Preprocessing_workflow.connect(aw_fmap_unmasked_epi, "out_file", fugue_shift, "fmap_in_file")
Preprocessing_workflow.connect(fieldmaprads2epi_bin,"out_file", fugue_shift, "mask_file") 

Preprocessing_workflow.connect(fugue_sbref, "unwarped_file", convert_fmap_shift, "reference")
Preprocessing_workflow.connect(fugue_shift,"shift_out_file", convert_fmap_shift, "shift_in_file")
Preprocessing_workflow.connect(flt_bbr, "out_matrix_file", convert_fmap_shift, "postmat")
Preprocessing_workflow.connect(motion_correct_epi, "mat_file", convert_fmap_shift, "premat")

Preprocessing_workflow.connect(file_list, "epi", split_epi, "in_file")

Preprocessing_workflow.connect(split_epi, "out_files", aw_final, "in_file") 
Preprocessing_workflow.connect(fugue_sbref, "unwarped_file", aw_final, "ref_file")
Preprocessing_workflow.connect(convert_fmap_shift, "out_file", aw_final, "field_file")

Preprocessing_workflow.connect(aw_final, "out_file", merge_epi, "in_files")

Preprocessing_workflow.connect(merge_epi, "merged_file" , epi_mean, "in_file")


Preprocessing_workflow.connect(strip_corrected_sbref, "out_file", flt_sbref_brain_t1_brain, "in_file")
Preprocessing_workflow.connect(T1_skull_strip, "BrainExtractionBrain", flt_sbref_brain_t1_brain, "reference")

Preprocessing_workflow.connect(fugue_sbref, "unwarped_file", flt_sbref_2_T1, "in_file")
Preprocessing_workflow.connect(n4, "output_image", flt_sbref_2_T1, "reference")
Preprocessing_workflow.connect(flt_sbref_brain_t1_brain, "out_matrix_file", flt_sbref_2_T1, "in_matrix_file")


Preprocessing_workflow.connect(strip_corrected_sbref, "out_file", bbr_sbref_2_T1, "in_file")  
Preprocessing_workflow.connect(n4, "output_image", bbr_sbref_2_T1, "reference")
Preprocessing_workflow.connect(fugue_dilate, "fmap_out_file", bbr_sbref_2_T1, "fieldmap") 
Preprocessing_workflow.connect(flt_sbref_2_T1, "out_matrix_file", bbr_sbref_2_T1, "in_matrix_file") 
Preprocessing_workflow.connect(T1_seg, "tissue_class_map", bbr_sbref_2_T1, "wm_seg")


Preprocessing_workflow.connect(n4, "output_image", Ants, "moving_image")

                              
Preprocessing_workflow.connect(Ants, "inverse_composite_transform", at, "transforms")                            
Preprocessing_workflow.connect(file_list, "t1", at, "reference_image")


Preprocessing_workflow.connect(bbr_sbref_2_T1, "out_matrix_file", invt_mat, "in_file")

Preprocessing_workflow.connect(at, "output_image", flt_parcels_2_sbref, "in_file")
Preprocessing_workflow.connect(invt_mat, "out_file", flt_parcels_2_sbref, "in_matrix_file")
Preprocessing_workflow.connect(fugue_sbref, "unwarped_file", flt_parcels_2_sbref, "reference")



Preprocessing_workflow.write_graph()
#Preprocessing_workflow.run()
Preprocessing_workflow.run(plugin=plugin, plugin_args=plugin_args)
#Preprocessing_workflow.run(plugin=plugin, plugin_args=plugin_args,updatehash=True)