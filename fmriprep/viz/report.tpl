<?xml version="1.0" encoding="utf-8" ?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<meta name="generator" content="Docutils 0.12: http://docutils.sourceforge.net/" />
<title></title>
<style type="text/css">
</style>
</head>
<body>



{% for sub_report in sub_reports %}
    <h2>{{ sub_report.name }}</h2>
    {% for elem in sub_report.elements %}
        {{ elem.name }}
        {% for image in elem.files_contents %}
            {{ image }}
        {% endfor %}
    {% endfor %}
{% endfor %}

</body>
</html>
