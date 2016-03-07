# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 14:32:13 2016

@author: craigmoodie
"""

import matplotlib as mpl
mpl.use('Agg')

from nipype.interfaces.utility import Function
from nipype.pipeline.engine import Workflow, Node, MapNode
from variables_reports import data_dir, work_dir, subject_list, plugin, plugin_args

import sys
sys.path.insert(0, '/home/cmoodie/qc_pipeline')

from twelve_image_report_function_w_error import generate_report

Report_workflow = Workflow(name="Report_workflow")
Report_workflow.base_dir = work_dir


################################### Selecting Files ################################################

from nipype import SelectFiles
templates = dict(fieldmap="Preprocessing_workdir/Final_preprocessing_workflow/_subject_id_{subject_id}/Fmap_Dilating/Topup_Fieldmap_rad_maths_fieldmap_unmasked_warp_fieldmap_unmasked.nii.gz",
                 fmap_mag="Preprocessing_workdir/Final_preprocessing_workflow/_subject_id_{subject_id}/TopUp/fieldmap_mag_topup_corrected.nii.gz",
                 fmap_mag_brain="Preprocessing_workdir/Final_preprocessing_workflow/_subject_id_{subject_id}/Fmap_Mag_BET/fieldmap_mag_topup_corrected_brain_mask.nii.gz",
                 raw_epi="heudiconv_dicom_conversion/Bids_Subjects/sub-{subject_id}/func/sub-{subject_id}_task-rest_acq-LR_run-1_bold.nii.gz",                                                                  
                 stripped_EPI="Preprocessing_workdir/Final_preprocessing_workflow/_subject_id_{subject_id}/EPI_BET/sub-{subject_id}_task-rest_acq-LR_run-1_bold_brain_mask.nii.gz",
                 corrected_epi_mean="Preprocessing_workdir/Final_preprocessing_workflow/_subject_id_{subject_id}/EPI_mean_volume/vol0000_warp_merged_mean.nii.gz",
                 #epi_mean_hr="Preprocessing_workdir/Final_preprocessing_workflow/_subject_id_{subject_id}/BBR_Epi_Mean_to_T1/bbr_epi_mean_2_T1.nii.gz",
                 #epi_mean_mni="Preprocessing_workdir/Final_preprocessing_workflow/_subject_id_{subject_id}/EPI_mean_volume/vol0000_warp_merged_mean.nii.gz",
                 sbref="heudiconv_dicom_conversion/Bids_Subjects/sub-{subject_id}/func/sub-{subject_id}_task-rest_acq-LR_run-1_sbref.nii.gz",
                 sbref_brain="Preprocessing_workdir/Final_preprocessing_workflow/_subject_id_{subject_id}/SBRef_BET/sub-{subject_id}_task-rest_acq-LR_run-1_sbref_brain_mask.nii.gz",
                 sbref_t1="Preprocessing_workdir/Final_preprocessing_workflow/_subject_id_{subject_id}/BBR_SBRef_to_T1/bbr_sbref_2_T1.nii.gz",
                 corrected_sbref="Preprocessing_workdir/Final_preprocessing_workflow/_subject_id_{subject_id}/SBRef_Unwarping/sub-{subject_id}_task-rest_acq-LR_run-1_sbref_unwarped.nii.gz",
                 t1="Preprocessing_workdir/Final_preprocessing_workflow/_subject_id_{subject_id}/Bias_Field_Correction/Bias_Corrected_T1.nii.gz",
                 t1_brain="Preprocessing_workdir/Final_preprocessing_workflow/_subject_id_{subject_id}/Ants_T1_Brain_Extraction/highres001_BrainExtractionMask.nii.gz",
                 t1_mni="Preprocessing_workdir/Final_preprocessing_workflow/_subject_id_{subject_id}/T1_2_MNI_Registration/ANTS_T1_2_MNI_Warped.nii.gz",
                 parcels_t1="Preprocessing_workdir/Final_preprocessing_workflow/_subject_id_{subject_id}/Apply_ANTS_transform_MNI_2_T1/Parcels_MNI_222_trans.nii.gz",
                 parcels_native="Preprocessing_workdir/Final_preprocessing_workflow/_subject_id_{subject_id}/Parcels_2_EPI_Mean_Affine_w_Inv_Mat/parcels_in_native_space.nii.gz")
file_list = Node(SelectFiles(templates), name = "File_Selection")
file_list.inputs.base_directory = data_dir
file_list.iterables = [("subject_id", subject_list)]
file_list.inputs.ignore_exception = True




################################### Defining plotting functions ################################################


def anatomical_overlay(in_file,overlay_file,out_file):
    
    import os.path
    import matplotlib as mpl
    mpl.use('Agg')
    from nilearn.plotting import plot_anat

    mask_display = plot_anat(in_file)
    mask_display.add_edges(overlay_file)
    mask_display.dim = -1
    #mask_display.add_contours(overlay_file)
    #mask_display.add_overlay(overlay_file)
    mask_display.title(out_file, x=0.01, y=0.99, size=15, color=None, bgcolor=None, alpha=1)
    
    mask_display.savefig(out_file)
    
    return os.path.abspath(out_file)
    
#anatomical_overlay(epi_mean,T1,anat_out_file)




def parcel_overlay(in_file,overlay_file,out_file):
    
    import os.path
    import matplotlib as mpl
    mpl.use('Agg')
    from nilearn.plotting import plot_epi

    mask_display = plot_epi(in_file)
    mask_display.add_edges(overlay_file)
    #mask_display.add_contours(overlay_file)
    #mask_display.add_overlay(overlay_file)
    mask_display.title(out_file, x=0.01, y=0.99, size=15, color=None, bgcolor=None, alpha=1)
    
    mask_display.savefig(out_file)
    
    return os.path.abspath(out_file)
    
#parcel_overlay(epi_mean,T1,out_file)




def stripped_brain_overlay(in_file,overlay_file,out_file):
    
    import os.path
    import matplotlib as mpl
    mpl.use('Agg')
    from nilearn.plotting import plot_roi

    mask_display = plot_roi(in_file,overlay_file,output_file=out_file,title=out_file,display_mode = "ortho", dim = -1)
    #mask_display.bg_img(overlay_file)
    #mask_display.title(out_file, x=0.01, y=0.99, size=15, color=None, bgcolor=None, alpha=1)
    #mask_display.display_mode = "yx"
    
    mask_display
    return os.path.abspath(out_file)
    
#stripped_brain_overlay(in_file,overlay_file,out_file)



################################### Condensing 4D Stacks into 3D mean volumes ################################################


from nipype.interfaces.fsl import MeanImage

fmap_mag_mean = Node(MeanImage(), name = "Fieldmap_mean")
fmap_mag_mean.inputs.output_type = "NIFTI_GZ"
fmap_mag_mean.inputs.dimension = 'T'

stripped_epi_mean = Node(MeanImage(), name = "Stripped_EPI_mean")
stripped_epi_mean.inputs.output_type = "NIFTI_GZ"
stripped_epi_mean.inputs.dimension = 'T'

######This isn't working!!!
raw_epi_mean = Node(MeanImage(), name = "Raw_EPI_mean")
raw_epi_mean.inputs.output_type = "NIFTI_GZ"
raw_epi_mean.inputs.dimension = 'T'
    
    
    
    
################################### Wrapping in Nipype Functions ################################################

    
    
fmap_overlay = Node(Function(input_names=["in_file","overlay_file","out_file"],
                            output_names=["out_file"],
                            function=parcel_overlay),name="Fieldmap_to_SBRef_Overlay")
fmap_overlay.inputs.out_file="Fieldmap_to_SBRef_Overlay.png"   



fmap_mag_BET = Node(Function(input_names=["in_file","overlay_file","out_file"],
                            output_names=["out_file"],
                            function=stripped_brain_overlay),name="Fieldmap_Mag_BET")
fmap_mag_BET.inputs.out_file="Fieldmap_Mag_BET.png" 



EPI_BET_report = Node(Function(input_names=["in_file","overlay_file","out_file"],
                            output_names=["out_file"],
                            function=stripped_brain_overlay),name="EPI_Skullstrip_Overlay")
EPI_BET_report.inputs.out_file="EPI_Skullstrip_Overlay.png"
 



sbref_unwarp_overlay = Node(Function(input_names=["in_file","overlay_file","out_file"],
                            output_names=["out_file"],
                            function=anatomical_overlay),name="SBRef_Unwarping_Overlay")
sbref_unwarp_overlay.inputs.out_file="SBRef_Unwarping_Overlay.png"   



epi_unwarp_overlay = Node(Function(input_names=["in_file","overlay_file","out_file"],
                            output_names=["out_file"],
                            function=anatomical_overlay),name="EPI_Unwarping_Overlay")
epi_unwarp_overlay.inputs.out_file="EPI_Unwarping_Overlay.png"  





SBRef_BET = Node(Function(input_names=["in_file","overlay_file","out_file"],
                            output_names=["out_file"],
                            function=stripped_brain_overlay),name="SBRef_BET")
SBRef_BET.inputs.out_file="SBRef_BET_Overlay.png" 


T1_SkullStrip = Node(Function(input_names=["in_file","overlay_file","out_file"],
                            output_names=["out_file"],
                            function=stripped_brain_overlay),name="T1_SkullStrip")
T1_SkullStrip.inputs.out_file="T1_SkullStrip_Overlay.png" 
 
    

parcels_2_EPI = Node(Function(input_names=["in_file","overlay_file","out_file"],
                            output_names=["out_file"],
                            function=anatomical_overlay),name="Parcels_to_EPI_Overlay")
parcels_2_EPI.inputs.out_file="Parcels_to_EPI_Overlay.png"


parcels_2_T1 = Node(Function(input_names=["in_file","overlay_file","out_file"],
                            output_names=["out_file"],
                            function=anatomical_overlay),name="Parcels_to_T1_Overlay")
parcels_2_T1.inputs.out_file="Parcels_to_T1_Overlay.png"


parcels_2_sbref = Node(Function(input_names=["in_file","overlay_file","out_file"],
                            output_names=["out_file"],
                            function=anatomical_overlay),name="Parcels_to_SBRef")
parcels_2_sbref.inputs.out_file="Parcels_to_SBRef_Overlay.png"  




epi_2_sbref = Node(Function(input_names=["in_file","overlay_file","out_file"],
                            output_names=["out_file"],
                            function=anatomical_overlay),name="Functional_to_SBRef_Overlay")
epi_2_sbref.inputs.out_file="Functional_to_SBRef_Overlay.png"


sbref_2_t1 = Node(Function(input_names=["in_file","overlay_file","out_file"],
                            output_names=["out_file"],
                            function=anatomical_overlay),name="SBRef_to_T1_Overlay")
sbref_2_t1.inputs.out_file="SBRef_to_T1_Overlay.png"


T1_2_MNI = Node(Function(input_names=["in_file","overlay_file","out_file"],
                            output_names=["out_file"],
                            function=anatomical_overlay),name="T1_to_MNI_Overlay")
T1_2_MNI.inputs.out_file="T1_to_MNI_Overlay.png"
T1_2_MNI.inputs.overlay_file="/share/sw/free/fsl/5.0.7/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz"




final_pdf = Node(Function(input_names=["output_file","first_plot","second_plot","third_plot","fourth_plot","fifth_plot","sixth_plot",
                            "seventh_plot","eighth_plot","ninth_plot","tenth_plot","eleventh_plot","twelfth_plot"],
                            output_names=["output_file"],
                            function=generate_report),name="Final_pdf")
final_pdf.inputs.output_file="Preprocessing_Quality_Report.pdf"



################################### Connecting workflow nodes ################################################


Report_workflow.connect(file_list, "fmap_mag", fmap_mag_mean, "in_file")
                            
Report_workflow.connect(file_list, "fieldmap", fmap_overlay, "in_file")
Report_workflow.connect(file_list, "sbref", fmap_overlay, "overlay_file")


Report_workflow.connect(file_list, "fmap_mag_brain", fmap_mag_BET, "in_file")
Report_workflow.connect(fmap_mag_mean, "out_file", fmap_mag_BET, "overlay_file")


Report_workflow.connect(file_list, "raw_epi", raw_epi_mean, "in_file")
Report_workflow.connect(file_list, "stripped_EPI", stripped_epi_mean, "in_file")


Report_workflow.connect(stripped_epi_mean, "out_file", EPI_BET_report,"in_file") 
Report_workflow.connect(raw_epi_mean, "out_file", EPI_BET_report, "overlay_file")


Report_workflow.connect(file_list, "corrected_sbref", sbref_unwarp_overlay, "in_file")
Report_workflow.connect(file_list, "sbref", sbref_unwarp_overlay, "overlay_file")

Report_workflow.connect(file_list, "sbref_brain", SBRef_BET, "in_file")
Report_workflow.connect(file_list, "sbref", SBRef_BET, "overlay_file")

Report_workflow.connect(file_list, "t1_brain", T1_SkullStrip, "in_file")
Report_workflow.connect(file_list, "t1", T1_SkullStrip, "overlay_file")

Report_workflow.connect(file_list, "parcels_native", parcels_2_EPI, "in_file")
Report_workflow.connect(file_list, "corrected_epi_mean", parcels_2_EPI, "overlay_file")

Report_workflow.connect(file_list, "parcels_t1", parcels_2_T1, "in_file")
Report_workflow.connect(file_list, "t1", parcels_2_T1, "overlay_file")

Report_workflow.connect(file_list, "parcels_native", parcels_2_sbref, "in_file")
Report_workflow.connect(file_list, "corrected_sbref", parcels_2_sbref, "overlay_file") ### prob should use corrected sbref brain mask

Report_workflow.connect(file_list, "corrected_epi_mean", epi_2_sbref, "in_file")
Report_workflow.connect(file_list, "corrected_sbref", epi_2_sbref, "overlay_file") 

Report_workflow.connect(file_list, "sbref_t1", sbref_2_t1, "in_file")
Report_workflow.connect(file_list, "t1", sbref_2_t1, "overlay_file")

Report_workflow.connect(file_list, "corrected_epi_mean", epi_unwarp_overlay, "in_file")
Report_workflow.connect(raw_epi_mean, "out_file", epi_unwarp_overlay, "overlay_file")

Report_workflow.connect(file_list, "t1_mni", T1_2_MNI, "in_file")


######################### replace sbref to mni and epi to mni with sbref and epi unwarping, also replace epi to t1 with parcel to sbref

Report_workflow.connect(fmap_overlay, "out_file", final_pdf, "first_plot")
Report_workflow.connect(EPI_BET_report, "out_file", final_pdf, "second_plot")
Report_workflow.connect(SBRef_BET, "out_file", final_pdf, "third_plot")
Report_workflow.connect(T1_SkullStrip, "out_file", final_pdf,  "fourth_plot")
Report_workflow.connect(sbref_unwarp_overlay, "out_file", final_pdf,  "fifth_plot")
Report_workflow.connect(epi_unwarp_overlay, "out_file", final_pdf,  "sixth_plot")
Report_workflow.connect(epi_2_sbref, "out_file", final_pdf,  "seventh_plot")
Report_workflow.connect(sbref_2_t1, "out_file", final_pdf,  "eighth_plot")
Report_workflow.connect(T1_2_MNI, "out_file", final_pdf,  "ninth_plot")
Report_workflow.connect(parcels_2_T1, "out_file", final_pdf,  "tenth_plot")
Report_workflow.connect(parcels_2_EPI, "out_file", final_pdf,  "eleventh_plot")
Report_workflow.connect(parcels_2_sbref, "out_file", final_pdf,  "twelfth_plot")

Report_workflow.write_graph()
#Report_workflow.run()
Report_workflow.run(plugin=plugin, plugin_args=plugin_args)
#Report_workflow.run(plugin=plugin, plugin_args=plugin_args,updatehash=True)

    