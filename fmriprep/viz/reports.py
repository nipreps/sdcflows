import json
import re
import os
from os.path import join, exists, basename, dirname

import matplotlib.image as mimage
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt

class Element(object):
    
    def __init__(self, name, file_pattern, title, description):
        self.name = name
        self.file_pattern = re.compile(file_pattern)
        self.title = title
        self.description = description
        self.files = []

class SubReport(object):

    def __init__(self, name, elements, **kwargs):
        self.name = name
        self.elements = []
        for e in elements:
            element = Element(**e)
            self.elements.append(element)
    
    def generate_sub_report_pdf(self):
        fig = plot.figure(figsize=(8.27, 11.69), dpi=300)
        return fig

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
                        if element.file_pattern.search(f):
                            element.files.append(f)

    def generate_report(self):
        full_report = PdfPages('report.pdf')
        for sub_report in self.sub_reports:
            full_report.savefiv(sub_report.generate_sub_report())
