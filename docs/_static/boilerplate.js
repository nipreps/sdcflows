function toggle() {

        var freesurfer_value = document.getElementById('freesurfer').checked;
        $("span[class^='freesurfer_text_']").each(function (i, el) {
          el.style.display='none'
     		});

        $("span[class='freesurfer_text_" + freesurfer_value + "']").each(function (i, el) {
          el.style.display='inline'
     		});

        var ss_template_value = document.getElementById('ss_template').value;
        $("span[class^='ss_template_text_']").each(function (i, el) {
          el.style.display='none'
     		});

        $("span[class='ss_template_text_" + ss_template_value + "']").each(function (i, el) {
          el.style.display='inline'
     		});

        var SDC_value = document.getElementById('SDC').value;
        $("span[class^='SDC_text_']").each(function (i, el) {
          el.style.display='none'
     		});

        $("span[class='SDC_text_" + SDC_value + "']").each(function (i, el) {
          el.style.display='inline'
     		});

        var freesurfer_value = document.getElementById('slicetime').checked;
        $("span[class^='slicetime_text_']").each(function (i, el) {
          el.style.display='none'
     		});

        $("span[class='slicetime_text_" + freesurfer_value + "']").each(function (i, el) {
          el.style.display='inline'
     		});

        var freesurfer_value = document.getElementById('AROMA').checked;
        $("span[class^='AROMA_text_']").each(function (i, el) {
          el.style.display='none'
     		});

        $("span[class='AROMA_text_" + freesurfer_value + "']").each(function (i, el) {
          el.style.display='inline'
     		});
        return false;
}