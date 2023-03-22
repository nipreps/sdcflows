Search.setIndex({docnames:["api","api/sdcflows.cli","api/sdcflows.cli.find_estimators","api/sdcflows.fieldmaps","api/sdcflows.interfaces","api/sdcflows.interfaces.brainmask","api/sdcflows.interfaces.bspline","api/sdcflows.interfaces.epi","api/sdcflows.interfaces.fmap","api/sdcflows.interfaces.reportlets","api/sdcflows.interfaces.utils","api/sdcflows.transform","api/sdcflows.utils","api/sdcflows.utils.bimap","api/sdcflows.utils.epimanip","api/sdcflows.utils.misc","api/sdcflows.utils.phasemanip","api/sdcflows.utils.tools","api/sdcflows.utils.wrangler","api/sdcflows.viz","api/sdcflows.viz.utils","api/sdcflows.workflows","api/sdcflows.workflows.ancillary","api/sdcflows.workflows.apply","api/sdcflows.workflows.apply.correction","api/sdcflows.workflows.apply.registration","api/sdcflows.workflows.base","api/sdcflows.workflows.fit","api/sdcflows.workflows.fit.fieldmap","api/sdcflows.workflows.fit.pepolar","api/sdcflows.workflows.fit.syn","api/sdcflows.workflows.outputs","changes","index","installation","links","methods"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":5,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.intersphinx":1,"sphinx.ext.viewcode":1,sphinx:56},filenames:["api.rst","api/sdcflows.cli.rst","api/sdcflows.cli.find_estimators.rst","api/sdcflows.fieldmaps.rst","api/sdcflows.interfaces.rst","api/sdcflows.interfaces.brainmask.rst","api/sdcflows.interfaces.bspline.rst","api/sdcflows.interfaces.epi.rst","api/sdcflows.interfaces.fmap.rst","api/sdcflows.interfaces.reportlets.rst","api/sdcflows.interfaces.utils.rst","api/sdcflows.transform.rst","api/sdcflows.utils.rst","api/sdcflows.utils.bimap.rst","api/sdcflows.utils.epimanip.rst","api/sdcflows.utils.misc.rst","api/sdcflows.utils.phasemanip.rst","api/sdcflows.utils.tools.rst","api/sdcflows.utils.wrangler.rst","api/sdcflows.viz.rst","api/sdcflows.viz.utils.rst","api/sdcflows.workflows.rst","api/sdcflows.workflows.ancillary.rst","api/sdcflows.workflows.apply.rst","api/sdcflows.workflows.apply.correction.rst","api/sdcflows.workflows.apply.registration.rst","api/sdcflows.workflows.base.rst","api/sdcflows.workflows.fit.rst","api/sdcflows.workflows.fit.fieldmap.rst","api/sdcflows.workflows.fit.pepolar.rst","api/sdcflows.workflows.fit.syn.rst","api/sdcflows.workflows.outputs.rst","changes.rst","index.rst","installation.rst","links.rst","methods.rst"],objects:{"sdcflows.cli":[[2,0,0,"-","find_estimators"]],"sdcflows.cli.find_estimators":[[2,1,1,"","gen_layout"],[2,1,1,"","main"]],"sdcflows.fieldmaps":[[3,2,1,"","EstimatorType"],[3,2,1,"","FieldmapEstimation"],[3,2,1,"","FieldmapFile"],[3,5,1,"","MetadataError"],[3,1,1,"","clear_registry"],[3,1,1,"","get_identifier"]],"sdcflows.fieldmaps.EstimatorType":[[3,3,1,"","ANAT"],[3,3,1,"","MAPPED"],[3,3,1,"","PEPOLAR"],[3,3,1,"","PHASEDIFF"],[3,3,1,"","UNKNOWN"]],"sdcflows.fieldmaps.FieldmapEstimation":[[3,3,1,"","bids_id"],[3,4,1,"","get_workflow"],[3,3,1,"","method"],[3,4,1,"","paths"],[3,3,1,"","sources"]],"sdcflows.fieldmaps.FieldmapFile":[[3,3,1,"","bids_root"],[3,4,1,"","check_path"],[3,3,1,"","entities"],[3,3,1,"","find_meta"],[3,3,1,"","metadata"],[3,3,1,"","path"],[3,3,1,"","suffix"]],"sdcflows.interfaces":[[5,0,0,"-","brainmask"],[6,0,0,"-","bspline"],[7,0,0,"-","epi"],[8,0,0,"-","fmap"],[9,0,0,"-","reportlets"],[10,0,0,"-","utils"]],"sdcflows.interfaces.brainmask":[[5,2,1,"","BinaryDilation"],[5,2,1,"","BrainExtraction"],[5,2,1,"","Union"]],"sdcflows.interfaces.bspline":[[6,2,1,"","ApplyCoeffsField"],[6,2,1,"","BSplineApprox"],[6,2,1,"","TOPUPCoeffReorient"],[6,2,1,"","TransformCoefficients"],[6,1,1,"","bspline_grid"]],"sdcflows.interfaces.epi":[[7,2,1,"","GetReadoutTime"],[7,2,1,"","SortPEBlips"]],"sdcflows.interfaces.fmap":[[8,2,1,"","CheckB0Units"],[8,2,1,"","DisplacementsField2Fieldmap"],[8,2,1,"","PhaseMap2rads"],[8,2,1,"","Phasediff2Fieldmap"],[8,2,1,"","SubtractPhases"]],"sdcflows.interfaces.reportlets":[[9,2,1,"","FieldmapReportlet"]],"sdcflows.interfaces.utils":[[10,2,1,"","ConvertWarp"],[10,2,1,"","DenoiseImage"],[10,2,1,"","Deoblique"],[10,2,1,"","Flatten"],[10,2,1,"","PadSlices"],[10,2,1,"","PositiveDirectionCosines"],[10,2,1,"","Reoblique"],[10,2,1,"","ReorientImageAndMetadata"],[10,2,1,"","UniformGrid"]],"sdcflows.transform":[[11,2,1,"","B0FieldTransform"],[11,1,1,"","disp_to_fmap"],[11,1,1,"","fmap_to_disp"],[11,1,1,"","grid_bspline_weights"]],"sdcflows.transform.B0FieldTransform":[[11,4,1,"","apply"],[11,3,1,"","coeffs"],[11,4,1,"","fit"],[11,3,1,"","mapped"],[11,4,1,"","to_displacements"],[11,3,1,"","xfm"]],"sdcflows.utils":[[13,0,0,"-","bimap"],[14,0,0,"-","epimanip"],[15,0,0,"-","misc"],[16,0,0,"-","phasemanip"],[17,0,0,"-","tools"],[18,0,0,"-","wrangler"]],"sdcflows.utils.bimap":[[13,2,1,"","EstimatorRegistry"],[13,2,1,"","bidict"]],"sdcflows.utils.bimap.EstimatorRegistry":[[13,4,1,"","get_key"],[13,6,1,"","sources"]],"sdcflows.utils.bimap.bidict":[[13,4,1,"","add"],[13,4,1,"","clear"]],"sdcflows.utils.epimanip":[[14,1,1,"","epi_mask"],[14,1,1,"","get_trt"]],"sdcflows.utils.misc":[[15,1,1,"","create_logger"],[15,1,1,"","front"],[15,1,1,"","get_free_mem"],[15,1,1,"","last"]],"sdcflows.utils.phasemanip":[[16,1,1,"","au2rads"],[16,1,1,"","delta_te"],[16,1,1,"","phdiff2fmap"],[16,1,1,"","subtract_phases"]],"sdcflows.utils.tools":[[17,1,1,"","brain_masker"],[17,1,1,"","ensure_positive_cosines"],[17,1,1,"","reorient_pedir"],[17,1,1,"","resample_to_zooms"]],"sdcflows.utils.wrangler":[[18,1,1,"","find_estimators"]],"sdcflows.viz":[[20,0,0,"-","utils"]],"sdcflows.viz.utils":[[20,1,1,"","coolwarm_transparent"],[20,1,1,"","plot_registration"]],"sdcflows.workflows":[[22,0,0,"-","ancillary"],[23,0,0,"-","apply"],[26,0,0,"-","base"],[27,0,0,"-","fit"],[31,0,0,"-","outputs"]],"sdcflows.workflows.ancillary":[[22,1,1,"","init_brainextraction_wf"]],"sdcflows.workflows.apply":[[24,0,0,"-","correction"],[25,0,0,"-","registration"]],"sdcflows.workflows.apply.correction":[[24,1,1,"","init_unwarp_wf"]],"sdcflows.workflows.apply.registration":[[25,1,1,"","init_coeff2epi_wf"]],"sdcflows.workflows.base":[[26,1,1,"","init_fmap_preproc_wf"]],"sdcflows.workflows.fit":[[28,0,0,"-","fieldmap"],[29,0,0,"-","pepolar"],[30,0,0,"-","syn"]],"sdcflows.workflows.fit.fieldmap":[[28,1,1,"","init_fmap_wf"],[28,1,1,"","init_magnitude_wf"],[28,1,1,"","init_phdiff_wf"]],"sdcflows.workflows.fit.pepolar":[[29,1,1,"","init_3dQwarp_wf"],[29,1,1,"","init_topup_wf"]],"sdcflows.workflows.fit.syn":[[30,1,1,"","init_syn_preprocessing_wf"],[30,1,1,"","init_syn_sdc_wf"],[30,1,1,"","match_histogram"]],"sdcflows.workflows.outputs":[[31,2,1,"","DerivativesDataSink"],[31,1,1,"","init_fmap_derivatives_wf"],[31,1,1,"","init_fmap_reports_wf"]],"sdcflows.workflows.outputs.DerivativesDataSink":[[31,3,1,"","out_path_base"]],sdcflows:[[1,0,0,"-","cli"],[3,0,0,"-","fieldmaps"],[4,0,0,"-","interfaces"],[11,0,0,"-","transform"],[12,0,0,"-","utils"],[19,0,0,"-","viz"],[21,0,0,"-","workflows"]]},objnames:{"0":["py","module","Python module"],"1":["py","function","Python function"],"2":["py","class","Python class"],"3":["py","attribute","Python attribute"],"4":["py","method","Python method"],"5":["py","exception","Python exception"],"6":["py","property","Python property"]},objtypes:{"0":"py:module","1":"py:function","2":"py:class","3":"py:attribute","4":"py:method","5":"py:exception","6":"py:property"},terms:{"0":[3,6,8,9,10,11,14,15,16,20,24,28,29,30,33,34,36],"00017":36,"00059":14,"00119341":14,"00246":16,"00336":36,"005":3,"00522":16,"006":3,"00746":3,"00768":16,"01":[3,6,18,33],"011":36,"0152472":36,"01_dir":3,"01_fieldmap":3,"01_phase2":3,"01_phasediff":3,"01_t1w":3,"03":36,"04":33,"05":[18,33,36],"05251":14,"07":34,"08":[33,36],"09":33,"0rc1":32,"1":[3,6,9,10,11,13,14,15,18,24,28,29,30,33,34,36],"10":[33,36],"100":32,"100185":18,"1002":36,"1006":36,"101006":18,"1016":36,"102":32,"10354":36,"104":32,"1054":36,"108":32,"1099":36,"11":[33,36],"1109":36,"1115":36,"1127":36,"114":32,"115":32,"1156":36,"116":32,"1166":36,"12":33,"120":32,"122":32,"127":[14,32],"128":32,"13":33,"130":32,"131":32,"132":32,"133":32,"137":32,"1371":36,"138":32,"139":36,"14":33,"142":32,"143":32,"149":32,"1492":36,"15":33,"150":32,"151":32,"152":32,"153":32,"154":32,"155":32,"156":32,"157":32,"159":32,"16":[6,33,34,36],"160":32,"161":32,"162":32,"164":32,"165":32,"166":32,"168":32,"17":[6,36],"170":32,"171":36,"172":32,"173":32,"174":32,"175":32,"176":32,"178":[32,36],"18":33,"180":32,"182":32,"184":32,"185":32,"188":32,"19":36,"190":32,"191":32,"1910340111":36,"193":36,"197":[32,36],"199":32,"1995":36,"1997":36,"199706":36,"1999":6,"1d":11,"2":[3,6,10,11,13,14,18,33,34,36],"20":[6,33,36],"200":32,"2000":36,"2001":36,"2002":36,"2003":36,"2004":36,"2014":36,"2016":36,"2017":36,"2019":33,"202":32,"2020":33,"2021":33,"2022":33,"2023":33,"20261":36,"204":32,"205":32,"207":32,"209":32,"21":32,"210":32,"212":32,"214":32,"215":[14,32],"216":32,"217":[32,36],"22":[6,33],"2227266":14,"223":32,"224":32,"228":32,"229":32,"23":33,"230":32,"234":32,"237":32,"239":32,"24":33,"240":[32,36],"243":32,"247":32,"248":32,"25":33,"251":32,"253":32,"254":32,"257":32,"258":32,"26":33,"261":32,"262":32,"263":32,"265":32,"269":32,"27":33,"270":32,"276":32,"277":32,"278":32,"28":[8,28,32],"280":32,"281":32,"282":32,"284":32,"285":32,"287":32,"288":32,"29":33,"292":32,"293":32,"298":32,"299":32,"2d":6,"2e6faa0":32,"2pi":32,"3":[3,5,6,10,11,13,14,17,18,30,33,34,36],"30":[20,33],"301":32,"302":32,"303":32,"304":32,"308":32,"309":32,"310":32,"311":32,"313":32,"314":32,"315":32,"317":32,"319":32,"321":32,"322":32,"324":32,"326":32,"329":32,"334":32,"335":32,"337":32,"3389":36,"339":32,"34":36,"3408":32,"341":32,"343":32,"35":14,"36":18,"361cd67":32,"38":6,"39":32,"3c171":36,"3d":[6,10,36],"3dqwarp":[10,29,32,36],"3e3":36,"3mm":30,"4":[3,6,10,13,14,18,33,36],"40":[6,15],"4096":8,"41":32,"42":36,"43":32,"434":14,"44":32,"449c2c2":32,"45":32,"450":36,"461":36,"47":32,"48":32,"49":[32,36],"4d":[10,32],"4mm":32,"5":[3,6,11,17,18,33,34,36],"50":[10,32],"51":32,"52":[32,36],"53":32,"55":32,"56":32,"57":32,"576":36,"58":32,"5f":[14,16],"6":[6,8,11,28,33,34,36],"60":32,"61":32,"63":32,"65":36,"7":[9,18,20,33,36],"70":32,"717a69":32,"73":36,"7325":14,"75":32,"79":32,"8":[20,33],"80":32,"8119":36,"814":36,"82":32,"870":36,"88":32,"888":36,"89":32,"896788":36,"9":[14,33,34],"90":32,"91":32,"95":32,"96":32,"97":32,"98":32,"abstract":9,"boolean":[6,8,9,10,31],"break":32,"byte":[10,18],"case":[6,18,25,32,36],"class":[0,3,5,6,7,8,9,10,11,13,31],"default":[5,6,8,9,10,11,20,22,26,28,31,32,36],"do":[14,32,36],"enum":3,"final":[11,18,32,33],"float":[5,6,7,8,9,11,30],"function":[0,11,17,18,30,32,34,36],"import":[17,18,32,34],"int":[11,15,22,24,25,26,28,29,30],"mika\u00ebl":32,"new":[13,17,33],"return":[3,11,13,15,17,18,32],"short":[28,29,30],"switch":32,"true":[3,6,8,10,11,13,17,18,30,31,36],"try":36,"universit\u00e4t":36,A:[3,6,9,10,11,13,18,26,28,29,30,31,32,34,36],As:[14,32,34],By:[11,36],For:[11,14,17,18,28,36],If:[10,11,14,18,30,36],In:[6,17,36],It:[14,36],NOT:32,No:24,Not:30,ONE:36,On:34,One:[18,31,36],Such:36,The:[3,5,6,8,10,11,14,18,25,26,28,29,30,31,32,33,34,36],Then:[14,36],There:36,These:[6,34,36],To:[10,32,36],With:[14,32],_:13,__version__:34,_bold:18,_dwi:18,_fieldmap:28,_sbref:18,_t1w:18,_t2w:18,about:[14,32],abov:[28,34],absenc:36,absolut:14,acc:14,acceler:[14,36],accept:[17,32],access:[32,33],accordingli:32,account:[14,17,24,36],accur:[28,29,30,32,36],accuraci:14,acquir:[33,36],acquisit:[14,28,33,36],across:[29,30,32],action:32,actual:[3,24,28],ad:[6,32],adapt:[6,32],add:[10,13,32],addit:[10,25,34],address:32,adebimp:32,adopt:32,advanc:32,affect:32,affili:32,affin:32,afni:[29,34,36],after:[5,6,10,11,18,22,24,28,29,32],afterward:32,aid:36,air:[17,28],al:[17,36],algorithm:[32,36],align:[6,25,32,36],all:[3,6,10,11,13,17,18,32,34],allevi:6,allow:[14,18,32],allowed_ent:31,along:[6,14,24,32,36],alpha:9,alreadi:[13,28,32,36],also:[18,24,30,32,36],alter:32,altern:[14,36],alwai:6,ambigu:32,amic:32,an:[5,6,7,8,9,10,11,13,14,17,18,24,25,26,28,29,30,31,32,33,34,36],analysi:[33,36],anat:[3,18],anat_mask:30,anat_nii:20,anat_ref:30,anatom:[18,25,28,30,31,36],ancillari:[0,21],andersson2003:36,andersson:36,ani:[7,8,10,31,36],annoi:32,anoth:[6,13],ant:[8,10,11,32,34,36],anterior:30,antsapplytransform:36,anyon:32,ap:18,apach:32,api:[32,33,36],appli:[0,11,18,21,28,29,30,32,36],applic:[33,36],apply_mask:9,applycoeffsfield:[6,32],approach:[3,32,33],appropri:[24,32],approx:14,approxim:[6,11,30,32],april:33,ar:[3,6,7,8,10,11,14,17,18,28,29,30,31,32,33,34,36],arbitrari:[16,17],arc:14,archiv:32,area:30,arg:[10,13],argument:[3,10,18,32,36],argv:2,around:[14,20,32],arrai:[11,17,36],art:33,artifact:[32,36],asr:6,assess:14,asset:6,associ:[3,11,30,36],assum:[30,36],atla:[30,32],atlas_threshold:30,attempt:[18,36],attribut:3,au2rad:16,august:33,auto:[20,32],auto_00000:[13,18],auto_00001:[13,18],auto_00002:[13,18],auto_00003:18,auto_00004:18,auto_00005:18,auto_00006:18,auto_00007:18,auto_00008:18,auto_00011:18,auto_00012:18,auto_:18,auto_bold_nss:30,autogener:32,autom:[3,36],automat:[13,18,30],avail:[3,7,14,18,34,36],averag:[28,30,32,36],avoid:[10,32],ax:[6,10,17,32,36],axcodes2ornt:17,axi:[10,36],axial:20,azeez:32,b0:[8,33],b0field:32,b0fieldidentifi:[3,18,26,31,32,33],b0fieldsourc:32,b0fieldtransform:11,b1:28,b:[6,11,13,17,18,22,24,28,29,30,32,36],b_0:[6,11,14,28,32,36],back:[18,28,32],background:20,bad:32,balaban:36,ball:5,base:[0,3,5,6,7,8,9,10,11,13,21,28,29,30,31,33,34,36],base_directori:31,basenam:32,basi:36,basic:[15,18],basil:32,batteri:31,becaus:[6,14,18,36],been:[14,18,24,32,36],befor:[11,32],behind:36,being:33,believ:14,below:[30,36],berlin:36,besid:28,best:6,bet:32,beta:11,better:[3,14,32],between:[6,36],beyond:6,bid:[2,3,7,8,14,16,18,28,31,32,33],bidict:13,bidirect:13,bids_dir:2,bids_filt:18,bids_fmap_id:31,bids_id:[3,18],bids_root:3,bidslayout:18,bimap:[0,12],binari:[5,10,28,34],binarydil:5,biom:36,bioscienc:32,bit:14,blair:32,blip:[7,32],blocksiz:32,blur:11,bodi:[6,11],boilerpl:32,bold:[18,30,32,36],boldref:32,bool:[3,11,18,24,25,26,28,29,30,31],border:11,both:[9,17,32,36],boundari:6,brain:[5,6,9,14,22,25,28,30,31,32,36],brain_mask:17,brainextract:5,brainextraction_wf:22,brainmask:[0,4,22,30],bring:32,bs_space:6,bspline:[0,4,32,36],bspline_dist:22,bspline_grid:6,bspline_weight:6,bsplineapprox:[6,32],bug:32,bugfix:32,build:[3,30,31,32,34,36],built:[18,36],c:[6,13,34],cach:[11,32],caclul:32,calcul:[7,8,11,14,16,22,24,26,28,30,32,36],call:[3,11,13,14,16,24,32],can:[3,10,11,14,17,18,28,34,36],candid:[18,36],cannot:13,capabl:10,cardin:[6,32],caution:36,cdot:[11,14,36],center:[6,14,32],central:32,chang:[10,18,32],channel:[9,36],character:36,cheapli:32,check:[10,32,34],check_hdr:31,check_path:3,checkb0unit:8,choic:32,choke:32,choos:32,christoph:32,ci:32,cieslak:32,circl:32,circleci:32,citizen:32,clean:18,cleanup:32,clear:[13,36],clear_registri:[3,18],cli:[0,32,33],clip:[22,36],close:11,cnr:32,co:[30,36],code:[6,10,24,25,28,29,30,32],codespel:32,coeff2epi:32,coeff2epi_wf:25,coeff:11,coeffici:[6,11,24,25,26,28,29,30,31,32,36],coerc:31,cognit:32,collat:32,collect:[3,32],colleg:32,collis:32,coloc:32,color:20,combin:[11,26],come:18,command:[10,34],comment:32,commit:32,compact:[11,36],compat:[10,11,32,36],complet:32,complic:[18,36],compon:36,comprehens:[18,32],compress:[20,31],comput:[10,11,30],concaten:32,concurr:32,conda:32,conduc:32,config:32,configur:[10,25,29,30],confus:14,connect:32,consecut:28,consid:32,consist:[3,7,10,32,36],consol:14,constant:11,constraint:32,consum:10,contain:[11,13,18,28,29,31,32,36],context:32,continu:36,contour:20,contrast:30,contribut:32,contributor:32,control:[6,11,25],control_zooms_mm:6,convent:7,convers:[29,30,32],convert:[6,8,10,11,16,28,32,36],convertwarp:10,coolwarm:20,coolwarm_transpar:20,coordin:[10,11,36],copi:10,copy_head:10,copyheaderinterfac:10,copyright:32,core:[5,6,7,8,10,28],coregistr:36,cornerston:36,coron:20,correct:[0,6,10,11,23,28,32,33,36],corrected_mask:24,correctli:[18,32],correspond:[3,7,8,10,18,24,25,28,29,30,32,33,36],cosin:[6,10,17],cost:[30,36],cover:32,coverag:32,cox1997:36,cox:36,cpu:[29,30,32],creat:[6,11,18,26,29,30,32],create_logg:15,ctrl_nii:11,cubic:[6,11],custom_ent:31,cut:20,cval:11,cyceron:32,d:[10,13,32],d_:36,data:[3,5,6,7,10,11,14,17,18,31,32,33],data_dtyp:31,database_dir:2,datalad:32,dataset:[10,18,24,25,29,31,32,33],datasink:31,datatyp:31,date:32,deal:[7,8],debian:34,debug:[6,24,25,26,28,29,30,32],decemb:33,decim:6,dedupl:32,deep:32,defin:[31,36],deform:[30,36],del:13,delta:[28,36],delta_:[14,16,36],delta_t:16,demean:[8,32],denoiseimag:10,dens:32,deobliqu:[10,36],depart:32,depend:[26,32,33],deploi:[32,34],deploy:32,deprec:32,dept:32,deriv:[26,31,32,33,36],derivativesdatasink:[31,32],describ:[10,32,36],descript:[28,29,30,32,34],design:[11,32,33],detach:32,detail:[32,36],detect:[30,32],determin:11,develop:[32,33,36],diagon:6,dict:[13,16,18,28,29,30,31],dictionari:[7,8,10,13,18,24,29,30,31],differ:[3,8,14,16,18,28,32,33],diffus:36,dilat:5,dimens:[6,10,11],dimension:[10,36],direct:[6,7,8,10,14,17,18,29,30,32,33],directli:[11,28,32,36],director:6,directori:[2,26,31],disabl:3,disappear:14,discov:[32,33],discuss:[14,36],dismiss:[11,18],dismiss_ent:31,disp_to_fmap:11,displac:[6,8,10,11,24,30,32,36],displacementsfield2fieldmap:8,displai:33,distanc:[22,32],distinct:32,distort:[6,11,24,25,30,32,33],distorted_ref:24,distortedref:32,distribut:6,div_id:20,divid:[32,36],dmriprep:32,doc:32,docker:32,dockerfil:34,docstr:32,document:[14,32],doe:14,doi:36,done:[14,28,32],dot:32,dotsb:[28,36],doubl:32,downsampl:32,downstream:32,dr:14,drift:36,driven:36,drop:32,ds000054:18,ds000206:18,ds001600:18,ds001771:[18,32],dsa:18,dsa_dir:3,dsb:18,dsc:18,dseg:32,dtype:[11,31],dure:32,dwi:[18,32,36],dynam:32,e0152472:36,e:[8,11,13,17,30,31,32],each:[6,11,24,29],earli:14,eas:36,easili:[11,33],echo:[14,28,33,36],echospac:14,echotim:[3,28],echotime1:[16,28],echotime2:[3,16,28],echotimediffer:16,ee:14,effect:[6,14,32,36],effectiveechospac:14,effici:32,effigi:32,eilidh:32,either:[6,28],elaps:36,element:[5,15,29,30],elsewher:28,embed:[6,11],emploi:10,empti:[3,10,13,18],enabl:[3,6,26,32],encod:[6,8,18,29,30,32,36],enforc:32,engin:14,enh:32,enhanc:[32,36],enough:14,ensur:[8,10,28,32],ensure_positive_cosin:17,enthusiast:36,entiti:[3,18,31,32],environ:[10,34],epi:[0,4,5,6,10,11,14,17,18,22,24,25,29,30,32,33,36],epi_mask:[14,30],epi_ref:30,epifactor:14,epimanip:[0,12],eq:[6,11,14,36],eqref:[6,11,14,36],equival:[14,28,36],error:[3,6,14,32,36],es:14,especi:36,esteban2016:36,esteban:[32,36],estim:[2,3,10,13,18,22,25,26,28,29,30,31,32,33],estimate_bright:20,estimatorregistri:13,estimatortyp:[3,18],et:36,etc:[14,32],evad:32,evalu:[6,11,32,36],even:10,everi:36,exampl:[3,13,15,16,18],except:[3,18],exclud:30,exclus:[10,18],execut:[10,18,30,36],exercis:32,exist:[5,6,7,8,9,10,31,36],expect:[10,30],experiment:36,explicit:33,exploit:36,expos:32,express:36,extend:11,extens:32,extent:11,extern:33,extra:[6,31],extract:[3,5,10,22,31,32],extrapol:[6,11,32],ey:32,f:[3,6,11,14,16,32],f_:14,fact:36,factor:[6,10,14,28,36],failur:32,fall:11,fals:[3,6,8,9,10,11,18,20,24,25,26,28,29,30,31],fast:[24,25,28,29,30,32,36],faster:26,fat:14,featur:32,februari:33,fed:[7,29,36],feed:6,feedback:36,few:18,fewer:36,fiddl:32,field:[3,6,8,10,11,14,18,24,25,26,28,29,30,31,32,36],fieldcoeff:[6,32],fieldmap:[0,6,7,8,9,11,13,16,18,22,24,25,26,27,29,30,31,32,33],fieldmap_ref:24,fieldmapestim:[3,13,18,26,32],fieldmapfil:[3,32],fieldmapreportlet:[9,32],fieldwarp:[24,32],fieldwarp_ref:24,fig:36,file1:13,file2:13,file3:13,file4:13,file5:13,file:[3,5,6,7,8,9,10,11,18,22,25,29,30,31,32,34],filenam:[3,9,10,31],filepath:3,filter:[6,11,18,32,36],find:[14,18,32,36],find_estim:[0,1,18,36],find_meta:3,fine:32,first:[10,14,29,30,32,36],fit:[0,6,11,21,32,36],fix:[31,32,36],fixat:6,fixed_hdr:31,fixhead:10,fixup:32,flag:3,flake8:32,flatten:[10,13,29],fledg:6,flexibil:32,flexibl:[32,36],flirt:32,float32:11,float64:11,flow:32,fm:18,fmap:[0,3,4,25,26,28,29,30],fmap_coeff:[24,25,26,28,29,30,31],fmap_derivatives_wf:31,fmap_mask:[25,28,29,31],fmap_nii:11,fmap_preproc_wf:26,fmap_ref:[6,25,26,28,29,30,31],fmap_reports_wf:31,fmap_target:6,fmap_to_disp:11,fmap_typ:31,fmap_wf:28,fmapid:32,fmapless:[18,36],fmri:36,fmriprep:[18,32],fninf:36,focus:32,folder:36,follow:[6,11,14,17,32,34,36],forc:[10,18,36],force_fmapless:[18,36],foreground:[20,32],fork:32,form:[6,10,11,17,32],formal:11,format:[11,24,36],formul:14,forward:32,found:[14,18,32,34,36],four:18,fov:32,frac:[14,36],framework:[32,36],franc:32,free:15,free_mem:24,freeli:36,freesurf:34,freie:36,frequenc:14,from:[3,6,7,8,10,14,15,16,17,18,22,24,28,30,31,32,34,36],from_fil:[5,6,7,8,10],front:[15,32,36],fsl:[6,7,14,28,29,32,34,36],full:[6,20],fulli:6,further:[14,36],g:[8,14,30,31,32],gamma:[14,36],gaussian:10,gen_layout:2,gener:[6,9,11,13,25,26,28,32],geometr:36,get:[13,14,32],get_data:32,get_fdata:32,get_free_mem:15,get_identifi:3,get_kei:13,get_trt:14,get_workflow:3,getreadouttim:7,gh:32,git:32,github:32,given:[3,8,11,16,18,22,24,29,30,32,36],goncalv:32,gorgolewski:32,gradient:36,graph:[24,25,28,29,30],grappa:14,grayscal:[14,17],gre:[5,22,28,36],grid:[6,11,17,24,29,32],grid_bspline_weight:11,grid_refer:29,group:17,guid:[14,28],guidelin:32,gyromagnet:[14,36],gz:[3,14,28],h:[32,36],ha:[6,11,14,17,18,24,32,36],habitu:34,handl:[32,34],hard:32,harvard:32,hashmap:13,have:[6,10,17,18,20,28,29,30,32,36],hcp101006:18,hcp:[18,32],head:[6,24],header:[6,10,31,32],hear:32,held:14,henc:6,here:[6,14,17,36],heurist:[18,32,36],hi:14,high:[32,36],higher:[32,36],histogram:[30,36],histori:32,hmc_xform:24,homogen:32,hospit:32,host:32,hot:32,hotfix:32,housekeep:32,how:[11,14,18,34,36],huntenburg2014:36,huntenburg:36,hutton2002:36,hutton:36,hz:[8,9,11,16,24,28,31,32,36],i:[6,7,8,10,11,14,17,36],idea:32,ident:[17,32],identifi:[3,11,18,32,36],ieee:[6,36],imag:[6,7,10,11,14,17,18,22,24,25,28,30,32,33],imagingfrequ:14,img:[6,14,17],img_mask:30,implement:[11,14,32,33],implicit:33,improv:32,in1:5,in2:5,in_:26,in_anat:30,in_coeff:6,in_data:[6,7,10,29],in_epi:[10,30],in_field:10,in_fil:[5,7,8,10,14,16,17,22,31],in_mask:[6,10],in_meta:[8,10,14,16,30],in_orient:10,in_phas:[8,16],in_plumb:10,in_valu:16,in_xfm:6,includ:[32,36],inclus:18,incomplet:32,inconsist:32,incorpor:32,increas:32,index:[3,10,29],indic:[3,10,18],individu:[24,25,26,28,36],infer:[3,10],inform:[0,32,36],infrastructur:32,inhomogen:[24,36],init_3dqwarp_wf:29,init_brainextraction_wf:22,init_coeff2epi_wf:25,init_fmap_derivatives_wf:31,init_fmap_preproc_wf:[26,32],init_fmap_reports_wf:31,init_fmap_wf:28,init_magnitude_wf:28,init_phdiff_wf:28,init_syn_preprocessing_wf:[30,36],init_syn_sdc_wf:[30,36],init_topup_wf:[29,32],init_unwarp_wf:24,initi:[3,18,32],inlin:32,inlist:15,input:[3,5,6,7,8,9,10,11,16,17,18,22,24,25,26,28,29,30,31,32,33,36],input_imag:10,insert:[13,32],instal:[32,33],instanc:[3,17,18,32,36],instead:32,inted:32,integ:[6,10,22,36],integr:[32,36],intend:10,intendedfor:[3,18,32,33],intens:[30,32,36],intensityclip:32,intent:36,interfac:[3,17,31,32,33,36],intern:[11,29,30],interpol:[11,24,32,36],interpret:[34,36],intersphinx:32,intramodalmerg:32,introduc:36,inu:32,invers:[11,17,30],invert:17,investig:18,ishandl:32,isol:32,issu:32,item:[6,7,8,10,13,31],iter:11,itk:[8,10,11,24],itk_format:11,itk_transform:8,its:[24,28,31,36],j:[6,7,8,10,14,17,32,36],januari:33,jenkinson2003:[28,36],jenkinson:36,jezzard1995:36,jezzard:36,journal:36,json:[3,14],juli:33,just:[6,22,28],k:[6,7,8,10,11,17,36],k_1:[6,11],k_2:[6,11],k_3:[6,11],keep:32,kei:[7,8,10,13,18,31,32],kernel:[11,32],kevin:32,keyerror:13,keyserv:32,keyword:3,king:32,knot:11,known:14,krzysztof:32,kwarg:[3,9,13],l:36,la:17,label:[6,9,11,14,18,20,26,32,36],laplacian:36,larg:[10,32],last:[3,13,14,15,16],latest:32,lausann:32,layout:18,least:36,left:[14,31,32],legaci:[32,36],legal:6,length:14,less:[18,26,28,29,30,32,33],lessen:10,let:36,level:[15,18,32,36],leverag:32,lfloor:14,librari:33,licens:34,lifespan:32,lighten:11,like:30,likewis:18,limit:[6,32],line:[10,34],linear:36,link:[14,32,36],list:[3,6,7,8,10,13,15,18,24,26,28,29,30,31],listserv:14,liter:32,literatur:36,load:32,local:32,locat:[6,11,31,36],log:[15,18,32],logger:[15,18],london:32,look:30,loosen:32,lower:18,lp:17,lr_epi:3,m:[6,34,36],macnicol:32,magazin:6,magn:36,magnet:14,magneticfieldstrength:14,magnitud:[22,25,28,31,32,36],magnitude1:18,magnitude2:18,magnitude_wf:28,mai:[3,6,14,24,25,26,28,33,36],main:2,maint:32,mainten:32,major:[32,36],make:[10,32,34,36],makefil:32,mandatori:[5,6,7,8,9,10,18,31],mani:[18,36],manifest:32,manipul:[14,16,32],manual:18,manuscript:32,map:[3,6,8,10,11,13,16,18,22,24,25,28,30,31,32,33],march:33,mark:6,markiewicz:32,marku:32,mask:[5,6,9,10,14,17,22,24,25,28,29,30,31,32],mask_anat:30,mass:32,massag:[28,32],master:36,match:[3,6,30],match_histogram:30,mathbf:[6,11],mathia:32,matric:[6,24,32],matrix:[11,14,32],matt:36,matter:14,matthew:32,max_alpha:[9,20],max_tr:10,maximum:[9,24,25,26,28,32],mean:[6,8,18,36],measur:28,mechan:14,med:36,median:6,medicin:32,memori:[15,32],merg:[30,32],messag:[18,32],meta:14,meta_dict:31,metadata:[3,7,8,10,11,14,16,18,24,28,29,30,31,32,36],metadataerror:3,method:[0,3,18,28,29,30,32,33],metric:30,mhz:14,middl:20,migrat:32,miniconda:32,minim:[32,33],minimum:[6,32],minor:32,mirror:11,misc:[0,12],miscellan:15,misconfigur:32,miss:18,mixin:[9,10],mm:30,mni152nlin2009casym:32,mni:30,mnt:32,mode:[6,11,24,25,28,29,30,32,36],model:[10,36],modifi:[6,20,32],modul:[0,1,4,12,19,21,23,27,32,33],more:[6,18,28,30,31,32,36],morpholog:[14,17],mosaic:9,most:[3,13,14,16,32,36],motion:[6,24],move:[6,9,25,32],moving_label:9,mr:[33,36],mri:[28,33,36],mrm:36,multi:[18,36],multipl:[29,32,36],multiplex:32,multivers:32,must:[18,29,34],mutual:[10,36],n4:[22,28,32],n:[10,11,36],n_:14,n_proc:32,name:[3,9,14,15,18,22,24,25,26,28,29,30,31,32],nan:32,natur:17,naveau:32,nbm453:36,ndarrai:11,nearest:11,necessari:[10,14,18,28,32,36],need:[28,32],neighbor:36,neither:14,neurodock:34,neuroimag:[32,33,34,36],neuroinform:36,neurostar:14,newest:32,newli:32,newpath:16,next:32,nibabel:[11,17],nibabi:32,niflow:32,nifti1imag:[6,11],nifti:[6,10,11,18,31,36],nii:[3,14,28],nimg:36,niprep:[32,33],nipyp:[5,6,7,8,9,10,31,32,33,34],nipype1:32,nitranform:32,niworkflow:[31,32],nmr:36,node:32,nois:[6,10,28],noise_imag:10,noise_model:10,nomin:32,non:36,nonbrain:22,none:[2,3,5,6,7,8,10,11,13,14,16,17,18,20,24,30,31,36],nonlinear:36,nonstandard:14,nonsteadi:30,nonuniform:28,nonzero:36,normal:[32,36],note:14,notebook:32,noth:3,notic:32,novemb:33,now:[15,32],num_thread:[6,10],number:[6,10,11,14,24,25,26,28,29,30,32,36],numpi:[11,31],nutshel:36,o:[10,36],obj:28,object:[3,5,6,7,8,9,10,11,18,31,32],obliqu:[6,10,32,36],obtain:[6,14,17,36],octob:33,odd:[6,32],oe:14,off:32,offer:33,offset:36,old:32,omp_nthread:[24,25,26,28,29,30],onc:14,one:[6,11,17,18,28,32,36],onli:[10,11,17,18,32,36],onto:36,opaque_perc:20,open:33,openneuro:18,openpgp:32,oper:[11,14,17,32],oppos:29,optim:[6,32,36],option:[3,5,6,7,8,9,10,11,14,18,22,26,29,30,31,32,36],orchestr:32,order:[11,17,20,32],org:32,orient:[6,17,32],origin:[6,10,11,31,32],ornt_transform:17,orthogon:29,os:[11,28,31],oscar:32,osf:32,oslo:32,other:[6,17,18,32,33,34],otherwis:15,our:[18,32],out:[11,31,32],out_:26,out_brain:22,out_coeff:6,out_correct:6,out_data:[7,10],out_epi:10,out_error:6,out_extrapol:6,out_field:[6,10],out_fil:[5,8,10,14,17,22,31],out_list:10,out_mask:[5,10,22],out_meta:[10,31],out_path_bas:31,out_probseg:[5,22],out_report:9,out_warp:[6,30,32],outcom:36,outlier:32,output:[0,5,6,7,8,9,10,11,21,22,24,25,26,28,29,30,32,36],output_dir:[26,31],output_dtyp:11,output_imag:10,outputnod:32,outsid:[6,9,11],over:[11,14,32],overflow:11,overhaul:32,overlai:20,overlay_param:20,overli:32,overload:31,overrid:32,overwrit:3,p:36,pa:[18,32],pacifi:32,packag:[0,32,33,34],pad:[10,17,32],padslic:10,page:32,pair:[13,18,28,32],parallel:[14,29,30,36],parallelreductionfactorinplan:14,paramet:[3,10,11,14,18,22,24,25,26,28,29,30,31,32,36],parekh:14,pars:2,part:[28,33],particip:[18,26],particular:[13,28],particularli:32,pass:[17,32,36],patch:[6,32],path:[3,6,11,15,28,29,30,31,32,34],pathlik:[5,6,7,8,9,10,11,28,31],pdf:[24,25,28,29,30],pe:[7,14,24,29,30,36],pe_dir:[6,7,8,10,11,17],pe_dir_fsl:7,pe_direct:7,pe_dirs_fsl:7,pennsylvania:32,pepolar4p:18,pepolar:[0,3,18,25,27,32,33],pepolar_estimate_wf:29,per:[11,14,18,28,32],perelman:32,perfect:6,perform:[3,17,25,36],permiss:32,phase1:[28,32],phase2:28,phase:[6,8,16,18,28,29,30,32,33],phase_diff:8,phasediff2fieldmap:8,phasediff:[3,18,28,32,36],phaseencodingdirect:[11,14,17,18],phasemanip:[0,12],phasemap2rad:8,phdiff2fmap:16,phdiff_wf:28,philip:14,physic:36,pi:[28,36],pick:[10,18],piec:32,piecewis:11,pin:32,pinsard:32,pip:34,pipelin:[32,36],pixel:[6,14],planar:[33,36],plane:14,platform:33,plausibl:36,pleas:[14,32],plo:36,plot:[20,32],plot_param:20,plot_registr:20,plumb:[6,10,32],png:[24,25,28,29,30],point:[6,11,36],polar:[10,17,32,36],poldrack:32,pone:36,pop:15,popul:32,posit:[6,10,17,36],positivedirectioncosin:10,possibl:[14,18,36],post:14,posterior:30,postprocess:32,potenti:[10,32],ppm:14,pravesh:14,pre:[33,36],precis:26,precomput:17,prefer:14,prefilt:[11,17],prelud:28,prepar:[28,30,32,36],prepend:11,preprocess:[14,26,28,30,31,32,33,36],present:18,prevent:32,previou:[11,32,36],print:[2,34],prioriti:18,probabilist:5,probabl:22,probe:15,problem:[6,14,32,36],procedur:36,process:[6,17,24,25,26,28,33,36],produc:36,product:[6,11,17,32],program:[10,32,33],project:[6,36],promin:[32,36],propag:31,properti:[13,17],proport:36,propos:36,protocol:[32,33,36],proton:36,provid:[3,11,14,18,28,32,33,36],psi:[6,11],psl:17,psycholog:32,pull:32,purpos:[18,25,36],push:32,put:32,pybid:[18,32],pypi:34,python:[32,33,34],q:6,qualiti:25,quantit:36,quick:[14,17],quit:14,r:[32,36],ra:[6,17,32],rad:[3,8,16,32],radian:[8,36],radiolog:32,radiu:5,rais:14,random:22,rang:[8,11,28,32,36],rather:[32,36],ratio:[14,36],re:32,reach:22,read:[3,16],readout:[7,8,11,14,24,36],readout_tim:7,reason:[6,32],receiv:[3,36],recent:[3,6,13,14,16,28],reconstruct:[14,36],ref_mask:30,refacor:32,refactor:32,refer:[6,9,10,11,18,22,24,25,26,28,29,30,31,32,33],reference_label:9,reflect:11,refresh:32,regard:14,region:[28,30,36],registr:[0,9,23,30,32,33],registri:[3,18,32],regress:8,regular:[6,30],rel:34,relai:18,relat:32,relax:32,releas:33,relev:32,reli:28,reliabl:32,remot:32,remov:[6,22,28,32],renam:32,render:32,reobliqu:10,reorient:[10,17,32],reorient_pedir:17,reorientimageandmetadata:10,repeat:13,replac:[22,32],repo:32,report:[9,14,31,32],reportcapableinterfac:9,reportlet:[0,4,32],repres:[3,5,6,7,8,9,10,11,31,36],represent:[11,32,36],reproduc:33,request:[17,32],requir:[28,33,34],resampl:[10,11,17,25,30,32],resample_to_zoom:17,research:36,resolut:32,resolv:[6,32],reson:36,resort:36,resourc:32,resource_monitor:[5,6,7,8,10],respect:17,restrict:[30,32],result:[11,32,36],retain:6,retouch:32,reus:[11,32],revers:[13,14,36],review:14,revis:[6,32],rf:32,rfloor:14,rician:10,ridge_alpha:6,right:[14,15,31],rigid:[6,11],ro:[14,36],ro_tim:[6,8,11],robust:[32,33,36],robustaverag:32,rodent:32,roll:32,root:[3,6,32],rorden:14,ross:32,rotat:[17,32,36],rotim:14,run:[10,18,24,25,28,29,30,32],runner:32,russel:32,s1053:36,s:[3,6,7,8,10,11,13,14,18,24,28,29,30,31,33,34,36],s_1:11,s_2:11,s_3:11,s_:36,sagitt:20,same:[10,32,36],sampl:[6,11],sar:6,satterthwait:32,save:[10,26,31,32],save_nois:10,sbref:[18,32],scalar:10,scale:[20,24,28,32],scan:[14,33,36],scanner:36,scheme:[33,36],school:32,scipi:32,scm:32,scrutin:36,sd:[30,36],sd_prior:30,sdc:[18,22,26,32,36],sdcflow:[0,34,36],search:[3,18,36],second:[11,14,30,36],secret:32,section:[11,36],see:[18,32,36],seem:14,segfault:[6,10],segment:[10,36],sei:[32,36],select:32,self:[13,32],sens:14,sensit:32,separ:[22,28,32],septemb:33,sequenc:[28,33],seri:33,serv:31,session:[18,32,36],set:[6,11,14,18,24,30,31,32,34,36],setup:32,setuptool:32,setuptools_scm:32,sever:[32,36],shape:[6,32],shift:[11,14,36],shim:36,should:[3,9,10,11,14,18,28,29,30,31,32,36],show:[9,32],shown:9,shrink:10,shrink_factor:10,sici:36,sidecar:[3,14],siemens2rad:32,signal:6,simpl:32,simplebeforeaft:32,simpleinterfac:[5,6,7,8,10],simplic:36,simplifi:[18,32],simplist:32,singl:[10,18,26,32,36],sitek:32,size:[14,24],sk:32,skim:32,skip:18,skull:[25,28,30,36],slice:[10,32,36],slightli:11,sloppi:[24,25,26,28,29,30,32],smaller:32,smart:22,smooth:[6,36],smriprep:32,sneve:32,so:[7,30,32,36],soft:22,softwar:[32,34,36],some:[3,14,18,22,28,32,34,36],somewher:15,sort:[3,7,22,32],sortpeblip:7,sourc:[2,3,5,6,7,8,9,10,11,13,14,15,16,17,18,20,22,24,25,26,28,29,30,31,33,36],source_fil:31,source_ornt:17,space:[6,11,14,24,25,30,31,32,36],spars:11,spatial:[10,11],spatialimag:[11,32],special:[9,13],specif:[0,3,32,33],specifi:[10,30],speech:32,sphinx:32,spike:6,spin:36,spinoza:14,spline:[6,11,22,24,28,29,30,32,36],spread:36,squeez:32,sra:6,stabil:32,stabl:32,stack:10,stage:36,standard:[10,14,32],stanford:32,start:32,stat:32,state:[30,33],statement:14,std2anat_xfm:30,step:[11,36],stop:32,store:[14,31,36],str:[3,10,11,15,18,22,24,25,26,28,29,30,31],strang:14,strategi:[3,6,26,32,36],strength:14,string:[5,6,7,8,9,10,31],strip:[25,28,30,36],strong:30,structur:[5,18,32,36],studholm:36,studholme2000:36,studi:36,sub:3,subject:[2,18,26,32],submit:32,submodul:[0,32,33,36],subordin:32,subpackag:[0,33],subsequ:[28,36],subtract:[28,32],subtract_phas:16,subtractphas:8,successfulli:18,suffer:36,suffic:18,suffici:32,suffix:3,suit:[32,36],sum_:6,summar:14,summari:36,support:[11,28,29,30,32],sure:[32,34],surfac:[11,36],surround:28,suscept:[11,30,32,33,36],svg:[9,24,25,28,29,30],svgutil:32,symmetr:36,symmetri:36,syn:[0,10,27,32,36],syn_preprocessing_wf:30,syn_sdc_wf:30,system:[10,34],t1:33,t1w:[3,30,36],t1w_invers:30,t2:30,t2w:[30,36],t:[14,16,36],t_:[14,36],t_mask:30,tabl:3,tag:[6,11,14,32,36],take:[28,29,36],taken:[24,29],target:[6,11,17,24,25,31,32,36],target_mask:25,target_nii:11,target_orient:10,target_ornt:17,target_ref:25,task:[29,30],te:[16,28,36],techniqu:33,technolog:32,templat:30,temporari:11,tensor:[6,11,32,36],tesla:36,test:[18,32],text:[14,16,28,36],than:32,thank:32,thei:[7,32,36],theodor:32,theori:33,therefor:[6,17,29,36],thereof:[11,25],thesi:36,theta:36,theta_i:36,thi:[3,6,8,10,11,14,17,18,24,25,26,28,29,30,32,33,36],third:13,thorough:14,those:36,thread:[6,10,24,25,26,28,32],three:36,threshold:30,through:[6,10,32],thu:17,time:[7,8,10,11,14,24,36],timepoint:30,tissu:22,to_displac:11,togeth:[30,32],toler:32,tomatch:31,tool:[0,12,20,32,33,34],topup:[6,7,10,29,32,36],topupcoeffreori:6,total:[7,8,11,14,36],totalreadouttim:[3,14],toward:32,tr:10,trace:25,traceback:[3,13,14,16],track:13,train:14,tran:36,transform:[0,6,8,17,30,33,36],transformcoeffici:6,transpar:[20,32],transparent_perc:20,travisci:32,treat:10,treiber2016:36,treiber:36,tri:[6,10],trigger:32,tune:32,tupl:[3,6,10,15,28,30],two:[14,16,17,18,28,29,32,36],txt:13,type:[3,7,8,11,13,18,31,36],typeerror:13,typic:[11,33,36],typo:32,u:[8,16],ubuntu:32,ucbn:32,um:32,unavil:32,uncompress:31,under:32,uneven:10,unhash:13,uniform:[32,36],uniformgrid:10,union:[5,18],uniqu:[3,24,25,28],unit:[3,8,16,24,28,31,32],unittest:32,univers:32,unknown:3,unless:11,unmodifi:31,unser1999:[6,11],unser:6,unsupport:33,untouch:15,unus:32,unwarp:[11,24,29,30,32,33],unwarp_wf:[24,32],unwrap:[28,32,36],up:[24,30,31,32,34,36],updat:[6,11,17,32,36],upgrad:32,upstream:32,url:32,us:[3,6,10,11,14,17,18,24,25,26,28,30,32,34,36],usa:32,usabl:32,user:[18,28,33],util:[0,3,4,19,32,33,36],v1:32,v:[10,36],val:13,valid:[3,11,14,18,32],valu:[3,5,6,7,8,9,10,11,13,18,31],valueerror:[3,13,14,16],variabl:10,variat:36,variou:[7,8],vein:32,veloc:36,vendor:14,verbos:[10,32],veri:18,version:[10,32,34],via:36,view:[20,32],vision:32,visual:[9,20,32,36],viz:[0,33],volum:[24,29,30,32],voxel:[6,11,22,24,30,36],voyag:14,vsm:[11,32,36],w:[14,32],wa:[10,11,28,29,30,31],wai:32,walker:22,wang2017:36,wang:36,want:[3,11,36],warn:32,warp:[10,30,36],water:14,waterfatshift:14,we:[3,11,18,28,36],weekli:32,weight:[11,33],well:[6,36],were:36,wf:14,what:33,when:[3,6,11,18,28,32,36],whenev:32,where:[6,9,11,14,30,36],whether:[8,24,28,29,30,31],which:[3,6,7,8,10,11,14,24,26,29,31,32,36],why:6,wire:32,within:[6,14,32,36],without:[10,30,32],wm:32,wonder:14,word:[6,17],work:32,workabl:36,workdir:32,workflow:[0,3,18,32,33,36],wrangler:[0,12,32,36],wrap:[8,10,11,28,36],write:[31,32],write_coeff:[25,31],written:[31,34],wrong:14,x:[6,7,10,20,33],xfm:[11,17],xyz_nii:11,y:[6,7,10,20,32],yield:36,your:34,z:[6,7,10,20,32],zaitsev2004:36,zaitsev:36,zero:[6,9,18,32],zoom:[6,17,32,36],zooms_min:6},titles:["Library API (application program interface)","sdcflows.cli package","sdcflows.cli.find_estimators module","sdcflows.fieldmaps module","sdcflows.interfaces package","sdcflows.interfaces.brainmask module","sdcflows.interfaces.bspline module","sdcflows.interfaces.epi module","sdcflows.interfaces.fmap module","sdcflows.interfaces.reportlets module","sdcflows.interfaces.utils module","sdcflows.transform module","sdcflows.utils package","sdcflows.utils.bimap module","sdcflows.utils.epimanip module","sdcflows.utils.misc module","sdcflows.utils.phasemanip module","sdcflows.utils.tools module","sdcflows.utils.wrangler module","sdcflows.viz package","sdcflows.viz.utils module","sdcflows.workflows package","sdcflows.workflows.ancillary module","sdcflows.workflows.apply package","sdcflows.workflows.apply.correction module","sdcflows.workflows.apply.registration module","sdcflows.workflows.base module","sdcflows.workflows.fit package","sdcflows.workflows.fit.fieldmap module","sdcflows.workflows.fit.pepolar module","sdcflows.workflows.fit.syn module","sdcflows.workflows.outputs module","What\u2019s new?","SDCFlows","Installation","&lt;no title&gt;","Methods and implementation"],titleterms:{"0":32,"01":32,"04":32,"05":32,"08":32,"09":32,"1":32,"10":32,"11":32,"12":32,"13":32,"14":32,"15":32,"16":32,"18":32,"2":32,"20":32,"2019":32,"2020":32,"2021":32,"2022":32,"2023":32,"22":32,"23":32,"24":32,"25":32,"26":32,"27":32,"29":32,"3":32,"30":32,"4":32,"5":32,"6":32,"7":32,"8":32,"9":32,"new":32,also:14,ancillari:22,api:0,appli:[23,24,25],applic:0,approach:36,april:32,august:32,author:32,b0:36,b0fieldidentifi:36,base:[26,32],bid:36,bimap:13,brainmask:5,bspline:6,cli:[1,2],content:33,correct:24,data:36,dataset:36,decemb:32,depend:34,differ:36,direct:36,discov:36,distort:36,epi:7,epimanip:14,estim:36,explicit:36,extern:34,februari:32,fieldmap:[3,28,36],find_estim:2,fit:[27,28,29,30],fmap:8,imag:36,implement:36,implicit:36,instal:34,intendedfor:36,interfac:[0,4,5,6,7,8,9,10],januari:32,juli:32,less:36,librari:0,list:32,mai:32,map:36,march:32,method:36,misc:15,modul:[2,3,5,6,7,8,9,10,11,13,14,15,16,17,18,20,22,24,25,26,28,29,30,31],novemb:32,octob:32,other:36,output:31,packag:[1,4,12,19,21,23,27],paper:32,pepolar:[29,36],phase:36,phasemanip:16,pre:32,program:0,refer:36,registr:[25,36],releas:32,reportlet:9,s:32,sdcflow:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33],see:14,septemb:32,sequenc:36,seri:32,specif:36,submodul:[1,4,12,19,21,23,27],subpackag:21,syn:30,t1:36,techniqu:36,thank:14,theori:36,tool:[17,36],transform:11,unsupport:36,unwarp:36,util:[10,12,13,14,15,16,17,18,20],viz:[19,20],weight:36,what:32,workflow:[21,22,23,24,25,26,27,28,29,30,31],wrangler:18,x:32}})