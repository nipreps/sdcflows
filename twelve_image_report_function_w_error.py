# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 16:54:08 2015

@author: craigmoodie
"""
def generate_report(output_file,first_plot,second_plot,third_plot,fourth_plot,fifth_plot,sixth_plot,seventh_plot,eighth_plot,ninth_plot,tenth_plot,eleventh_plot,twelfth_plot):
    
    import os.path
    import pylab as plt
    import nibabel as nb
    from matplotlib.gridspec import GridSpec
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.image as mimage
    #import python-markdown
    #import xhtml2pdf
    #import Markdown2PDF

    #error = mimage.imread("/scratch/PI/russpold/work/AA_Connectivity/error_image.png")
    error = mimage.imread("/Users/craigmoodie/Documents/AA_Connectivity_Psychosis/AA_Connectivity_Analysis/CNR.png")
    
    if first_plot is not None:
        #if os.path.exists(first_plot) :
            img1 = mimage.imread(first_plot)
        #else : img1 = error
    else : img1 = error
    
    if second_plot is not None:
        #if os.path.exists(second_plot) :
            img2 = mimage.imread(second_plot)
        #else : img2 = error
    else : img2 = error
    
    if fourth_plot is not None:
        #if os.path.exists(third_plot) :
            img3 = mimage.imread(third_plot)
        #else : img3 = error
    else : img3 = error
    
    if fifth_plot is not None:
        #if os.path.exists(fourth_plot) :
            img4 = mimage.imread(fourth_plot)
        #else : img4 = error
    else : img4 = error
    
    if sixth_plot is not None:    
        #if os.path.exists(fifth_plot) :
            img5 = mimage.imread(fifth_plot)
        #else : img5 = error
    else : img5 = error
    
    if seventh_plot is not None:
        #if os.path.exists(sixth_plot) :
            img6 = mimage.imread(sixth_plot)
        #else : img6 = error
    else : img6 = error
    
    if seventh_plot is not None:
        #if os.path.exists(seventh_plot) :
            img7 = mimage.imread(seventh_plot)
        #else : img7 = error
    else : img7 = error
    
    if eighth_plot is not None:
        #if os.path.exists(eighth_plot) :
            img8 = mimage.imread(eighth_plot)
        #else : img8 = error
    else : img8 = error
    
    if ninth_plot is not None:
        #if os.path.exists(eighth_plot) :
            img9 = mimage.imread(ninth_plot)
        #else : img9 = error
    else : img9 = error
    
    if tenth_plot is not None:
        #if os.path.exists(eighth_plot) :
            img10 = mimage.imread(tenth_plot)
        #else : img10 = error
    else : img10 = error
    
    if eleventh_plot is not None:
        #if os.path.exists(eighth_plot) :
            img11 = mimage.imread(eleventh_plot)
        #else : img11 = error
    else : img11 = error
    
    if twelfth_plot is not None:
        #if os.path.exists(eighth_plot) :
            img12 = mimage.imread(twelfth_plot)
        #else : img12 = error
    else : img12 = error

    
    report = PdfPages(output_file)

    fig = plt.figure()

    grid = GridSpec(4,3)

    ax = plt.subplot(grid[0,0])
    ax2 = plt.subplot(grid[1,0])
    ax3 = plt.subplot(grid[2,0])
    ax4 = plt.subplot(grid[3,0])
    ax5 = plt.subplot(grid[0,1])
    ax6 = plt.subplot(grid[1,1])
    ax7 = plt.subplot(grid[2,1])
    ax8 = plt.subplot(grid[3,1])
    ax9 = plt.subplot(grid[0,2])
    ax10 = plt.subplot(grid[1,2])
    ax11 = plt.subplot(grid[2,2])
    ax12 = plt.subplot(grid[3,2])
    

    

#    ax.plot(first_plot)  
#    ax2.plot(second_plot)
#    ax3.plot(third_plot)
#    ax4.plot(fourth_plot)


    ax.imshow(img1)
    ax2.imshow(img2)
    ax3.imshow(img3)
    ax4.imshow(img4)
    ax5.imshow(img5)
    ax6.imshow(img6)
    ax7.imshow(img7)
    ax8.imshow(img8)
    ax9.imshow(img9)
    ax10.imshow(img10)
    ax11.imshow(img11)
    ax12.imshow(img12)

    #ax.set_xlim((0, len(img1)))
    #ylim = ax.get_ylim()
    #ax.set_ylabel("Random Distribution 1")
    #ax.set_xlabel("Index 1")
    #ax.set_title(first_plot)
    ax.set_aspect('auto')    
    
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(8)
    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
        
        

    #ax2.set_xlim((0, len(img2)))
    #ylim = ax2.get_ylim()
    #ax2.set_ylabel("Random Distribution 2")
    #ax2.set_xlabel("Index 2")
    #ax2.set_title(os.path.split(second_plot))
    #ax.set_title(second_plot) 
    ax2.set_aspect('auto')   
    
    for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] +
             ax2.get_xticklabels() + ax2.get_yticklabels()):
        item.set_fontsize(8)
        
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)

    #ax3.set_xlim((0, len(img3)))
    #ylim = ax3.get_ylim()
    #ax3.set_ylabel("Random Distribution 3")
    #ax3.set_xlabel("Index 3")
    #ax3.set_title(third_plot)
    ax3.set_aspect('auto') 
    
    for item in ([ax3.title, ax3.xaxis.label, ax3.yaxis.label] +
             ax3.get_xticklabels() + ax3.get_yticklabels()):
        item.set_fontsize(8)
        
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)

    #ax4.set_ylabel("Random Distribution 4")
    #ax4.set_xlabel("Index 4")
    #ax4.set_title(fourth_plot)
    ax4.set_aspect('auto') 
    
    for item in ([ax4.title, ax4.xaxis.label, ax4.yaxis.label] +
             ax4.get_xticklabels() + ax4.get_yticklabels()):
        item.set_fontsize(8)
        
    ax4.get_xaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)
    
    #ax5.set_ylabel("Random Distribution 4")
    #ax5.set_xlabel("Index 4")
    #ax5.set_title(fifth_plot)
    ax5.set_aspect('auto') 
    
    for item in ([ax5.title, ax5.xaxis.label, ax5.yaxis.label] +
             ax5.get_xticklabels() + ax5.get_yticklabels()):
        item.set_fontsize(8)
        
    ax5.get_xaxis().set_visible(False)
    ax5.get_yaxis().set_visible(False)

    #ax6.set_ylabel("Random Distribution 4")
    #ax6.set_xlabel("Index 4")
    #ax6.set_title(sixth_plot)
    ax6.set_aspect('auto') 
    
    for item in ([ax6.title, ax6.xaxis.label, ax6.yaxis.label] +
             ax6.get_xticklabels() + ax6.get_yticklabels()):
        item.set_fontsize(8)
        
    ax6.get_xaxis().set_visible(False)
    ax6.get_yaxis().set_visible(False)
    
    #ax7.set_ylabel("Random Distribution 4")
    #ax7.set_xlabel("Index 4")
    #ax7.set_title(seventh_plot)
    ax7.set_aspect('auto') 
    
    for item in ([ax7.title, ax7.xaxis.label, ax7.yaxis.label] +
             ax7.get_xticklabels() + ax7.get_yticklabels()):
        item.set_fontsize(8)

    ax7.get_xaxis().set_visible(False)
    ax7.get_yaxis().set_visible(False)

    #ax8.set_ylabel("Random Distribution 4")
    #ax8.set_xlabel("Index 4")
    #ax8.set_title(eighth_plot)
    ax8.set_aspect('auto') 

    for item in ([ax8.title, ax8.xaxis.label, ax8.yaxis.label] +
             ax8.get_xticklabels() + ax8.get_yticklabels()):
        item.set_fontsize(8)

    ax8.get_xaxis().set_visible(False)
    ax8.get_yaxis().set_visible(False)
    
    #ax9.set_ylabel("Random Distribution 4")
    #ax9.set_xlabel("Index 4")
    #ax9.set_title(ninth_plot)
    ax9.set_aspect('auto') 

    for item in ([ax9.title, ax9.xaxis.label, ax9.yaxis.label] +
             ax9.get_xticklabels() + ax9.get_yticklabels()):
        item.set_fontsize(8)

    ax9.get_xaxis().set_visible(False)
    ax9.get_yaxis().set_visible(False)
    
    #ax10.set_ylabel("Random Distribution 4")
    #ax10.set_xlabel("Index 4")
    #ax10.set_title(tenth_plot)
    ax10.set_aspect('auto') 

    for item in ([ax10.title, ax10.xaxis.label, ax10.yaxis.label] +
             ax10.get_xticklabels() + ax10.get_yticklabels()):
        item.set_fontsize(8)

    ax10.get_xaxis().set_visible(False)
    ax10.get_yaxis().set_visible(False)


    #ax11.set_ylabel("Random Distribution 4")
    #ax11.set_xlabel("Index 4")
    #ax11.set_title(eleventh_plot)
    ax11.set_aspect('auto') 

    for item in ([ax11.title, ax11.xaxis.label, ax11.yaxis.label] +
             ax11.get_xticklabels() + ax11.get_yticklabels()):
        item.set_fontsize(8)

    ax11.get_xaxis().set_visible(False)
    ax11.get_yaxis().set_visible(False)
    
    
    #ax12.set_ylabel("Random Distribution 4")
    #ax12.set_xlabel("Index 4")
    #ax12.set_title(twelfth_plot)
    ax12.set_aspect('auto') 

    for item in ([ax12.title, ax12.xaxis.label, ax12.yaxis.label] +
             ax12.get_xticklabels() + ax12.get_yticklabels()):
        item.set_fontsize(8)

    ax12.get_xaxis().set_visible(False)
    ax12.get_yaxis().set_visible(False)
    
    
 
    fig.subplots_adjust(wspace=0.02, hspace=.2)

    report.savefig(fig, dpi=300)
    report.close()
    
    return os.path.abspath(output_file)
    
generate_report(output_file="report_generated.pdf", first_plot="/Users/craigmoodie/Documents/AA_Connectivity_Psychosis/AA_Connectivity_Analysis/Python_Scripts/EPI_2_T1_epi_function.png", 
                second_plot=None, 
                third_plot="/Users/craigmoodie/Documents/AA_Connectivity_Psychosis/AA_Connectivity_Analysis/Python_Scripts/SBRef_Skull_Stripping_2.png", 
                fourth_plot="/Users/craigmoodie/Documents/AA_Connectivity_Psychosis/AA_Connectivity_Analysis/Python_Scripts/EPI_2_T1_epi_function.png",
                fifth_plot="/Users/craigmoodie/Documents/AA_Connectivity_Psychosis/AA_Connectivity_Analysis/Python_Scripts/EPI_2_T1_epi_function.png",
                sixth_plot="/Users/craigmoodie/Documents/AA_Connectivity_Psychosis/AA_Connectivity_Analysis/Python_Scripts/EPI_2_T1_epi_function.png",
                seventh_plot="/Users/craigmoodie/Documents/AA_Connectivity_Psychosis/AA_Connectivity_Analysis/Python_Scripts/EPI_2_T1_epi_function.png",
                eighth_plot=None,
                ninth_plot="/Users/craigmoodie/Documents/AA_Connectivity_Psychosis/AA_Connectivity_Analysis/Python_Scripts/SBRef_Skull_Stripping_2.png",
                tenth_plot=None,
                eleventh_plot="/Users/craigmoodie/Documents/AA_Connectivity_Psychosis/AA_Connectivity_Analysis/Python_Scripts/SBRef_Skull_Stripping_2.png",
                twelfth_plot=None)