from __future__ import unicode_literals

import codecs
import json
import re
import os
from os.path import join, exists, basename, dirname

import jinja2
from pkg_resources import resource_filename as pkgrf

class Element(object):
    
    def __init__(self, name, file_pattern, title, description):
        self.name = name
        self.file_pattern = re.compile(file_pattern)
        self.title = title
        self.description = description
        self.files = []
        self.files_contents = []


class SubReport(object):

    def __init__(self, name, elements, **kwargs):
        self.name = name
        self.elements = []
        for e in elements:
            element = Element(**e)
            self.elements.append(element)

    def generate_sub_report(self, report):
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(searchpath='/'),
            trim_blocks=True, lstrip_blocks=True
        )
        sub_report_tpl = env.get_template('{}_tpl.html'.format(self.name))
        sub_report_render = sub_report_tpl.render
        return sub_report_render

class Report(object):
    
    def __init__(self, path, config):
        self.root = path
        self.sub_reports = []
        self._load_config(config)

    def _load_config(self, config):
        if isinstance(config, str):
            config = json.load(open(config, 'r'))

        for e in config['sub_reports']:
            sub_report = SubReport(**e)
            self.sub_reports.append(sub_report)

        self.index()

    def index(self):
        for root, directories, filenames in os.walk(self.root):
            for f in filenames:
                f = join(root, f)
                for sub_report in self.sub_reports:
                    for element in sub_report.elements:
                        if element.file_pattern.search(f) and f.split('.')[-1] == 'svg':
                            element.files.append(f)
                            with open(f) as fp:
                                content = fp.read()
                                content = '\n'.join(content.split('\n')[1:])
                                element.files_contents.append(content)

    def generate_report(self):
        searchpath = pkgrf('fmriprep', '/')
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(searchpath=searchpath),
            trim_blocks=True, lstrip_blocks=True
        )
        report_tpl = env.get_template('viz/report.tpl')
        report_render = report_tpl.render(sub_reports=self.sub_reports)
        return report_render
