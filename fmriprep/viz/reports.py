#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
fMRIprep reports builder
^^^^^^^^^^^^^^^^^^^^^^^^


"""

import os
from pathlib import Path
import json
import re

import html

import jinja2
from niworkflows.nipype.utils.filemanip import loadcrash, copyfile
from pkg_resources import resource_filename as pkgrf


class Element(object):
    """
    Just a basic component of a report
    """
    def __init__(self, name, title=None):
        self.name = name
        self.title = title


class Reportlet(Element):
    """
    A reportlet has title, description and a list of graphical components
    """
    def __init__(self, name, file_pattern=None, title=None, description=None, raw=False):
        self.name = name
        self.file_pattern = re.compile(file_pattern)
        self.title = title
        self.description = description
        self.source_files = []
        self.contents = []
        self.raw = raw


class SubReport(Element):
    """
    SubReports are sections within a Report
    """
    def __init__(self, name, reportlets=None, title=''):
        self.name = name
        self.title = title
        self.reportlets = []
        if reportlets:
            self.reportlets += reportlets
        self.isnested = False


class Report(object):
    """
    The full report object
    """
    def __init__(self, path, config, out_dir, run_uuid, out_filename='report.html'):
        self.root = path
        self.sections = []
        self.errors = []
        self.out_dir = out_dir
        self.out_filename = out_filename
        self.run_uuid = run_uuid

        self._load_config(config)

    def _load_config(self, config):
        with open(config, 'r') as configfh:
            config = json.load(configfh)

        self.index(config['sections'])

    def index(self, config):
        fig_dir = 'figures'
        subject_dir = self.root.split('/')[-1]
        subject = re.search('^(?P<subject_id>sub-[a-zA-Z0-9]+)$', subject_dir).group()
        svg_dir = os.path.join(self.out_dir, 'fmriprep', subject, fig_dir)
        os.makedirs(svg_dir, exist_ok=True)

        reportlet_list = list(sorted([str(f) for f in Path(self.root).glob('**/*.*')]))

        for subrep_cfg in config:
            reportlets = []
            for reportlet_cfg in subrep_cfg['reportlets']:
                rlet = Reportlet(**reportlet_cfg)
                for src in reportlet_list:
                    ext = src.split('.')[-1]
                    if rlet.file_pattern.search(src):
                        contents = None
                        if ext == 'html':
                            with open(src) as fp:
                                contents = fp.read().strip()
                        elif ext == 'svg':
                            fbase = os.path.basename(src)
                            copyfile(src, os.path.join(svg_dir, fbase),
                                     copy=True, use_hardlink=True)
                            contents = os.path.join(subject, fig_dir, fbase)

                        if contents:
                            rlet.source_files.append(src)
                            rlet.contents.append(contents)

                if rlet.source_files:
                    reportlets.append(rlet)

            if reportlets:
                sub_report = SubReport(
                    subrep_cfg['name'], reportlets=reportlets,
                    title=subrep_cfg.get('title'))
                self.sections.append(order_by_run(sub_report))

        error_dir = os.path.join(self.out_dir, "fmriprep", subject, 'log', self.run_uuid)
        if os.path.isdir(error_dir):
            self.index_error_dir(error_dir)

    def index_error_dir(self, error_dir):
        ''' Crawl subjects crash directory for the corresponding run and return text for
            .pklz crash file found. '''
        for root, directories, filenames in os.walk(error_dir):
            for f in filenames:
                crashtype = os.path.splitext(f)[1]
                if f[:5] == 'crash' and crashtype == '.pklz':
                    self.errors.append(self._read_pkl(os.path.join(root, f)))
                elif f[:5] == 'crash' and crashtype == '.txt':
                    self.errors.append(self._read_txt(os.path.join(root, f)))

    @staticmethod
    def _read_pkl(fname):
        crash_data = loadcrash(fname)
        data = {'file': fname,
                'traceback': ''.join(crash_data['traceback']).replace("\\n", "<br \>")}
        if 'node' in crash_data:
            data['node'] = crash_data['node']
            if data['node'].base_dir:
                data['node_dir'] = data['node'].output_dir()
            else:
                data['node_dir'] = "Node crashed before execution"
            data['inputs'] = sorted(data['node'].inputs.trait_get().items())
        return data

    @staticmethod
    def _read_txt(fname):
        with open(fname, 'r') as fobj:
            crash_data = fobj.read()
        lines = crash_data.splitlines()
        data = {'file': fname}
        traceback_start = 0
        if lines[0].startswith('Node'):
            data['node'] = lines[0].split(': ', 1)[1]
            data['node_dir'] = lines[1].split(': ', 1)[1]
            inputs = []
            for i, line in enumerate(lines[5:], 5):
                if not line:
                    traceback_start = i + 1
                    break
                inputs.append(tuple(map(html.escape, line.split(' = ', 1))))
            data['inputs'] = sorted(inputs)
        else:
            data['node_dir'] = "Node crashed before execution"
        data['traceback'] = '\n'.join(lines[traceback_start:])
        return data

    def generate_report(self):
        searchpath = pkgrf('fmriprep', '/')
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(searchpath=searchpath),
            trim_blocks=True, lstrip_blocks=True
        )
        report_tpl = env.get_template('viz/report.tpl')
        report_render = report_tpl.render(sections=self.sections, errors=self.errors)
        with open(os.path.join(self.out_dir, "fmriprep", self.out_filename), 'w') as fp:
            fp.write(report_render)
        return len(self.errors)


def order_by_run(subreport):
    ordered = []
    run_reps = {}

    for element in subreport.reportlets:
        if len(element.source_files) == 1 and element.source_files[0]:
            ordered.append(element)
            continue

        for filename, file_contents in zip(element.source_files, element.contents):
            name, title = generate_name_title(filename)
            if not filename or not name:
                continue

            new_element = Reportlet(
                name=element.name, title=element.title, file_pattern=element.file_pattern,
                description=element.description, raw=element.raw)
            new_element.contents.append(file_contents)
            new_element.source_files.append(filename)

            if name not in run_reps:
                run_reps[name] = SubReport(name, title=title)

            run_reps[name].reportlets.append(new_element)

    if run_reps:
        keys = list(sorted(run_reps.keys()))
        for key in keys:
            ordered.append(run_reps[key])
        subreport.isnested = True

    subreport.reportlets = ordered
    return subreport


def generate_name_title(filename):
    fname = os.path.basename(filename)
    expr = re.compile('^sub-(?P<subject_id>[a-zA-Z0-9]+)(_ses-(?P<session_id>[a-zA-Z0-9]+))?'
                      '(_task-(?P<task_id>[a-zA-Z0-9]+))?(_acq-(?P<acq_id>[a-zA-Z0-9]+))?'
                      '(_rec-(?P<rec_id>[a-zA-Z0-9]+))?(_run-(?P<run_id>[a-zA-Z0-9]+))?')
    outputs = expr.search(fname)
    if outputs:
        outputs = outputs.groupdict()
    else:
        return None, None

    name = '{session}{task}{acq}{rec}{run}'.format(
        session="_ses-" + outputs['session_id'] if outputs['session_id'] else '',
        task="_task-" + outputs['task_id'] if outputs['task_id'] else '',
        acq="_acq-" + outputs['acq_id'] if outputs['acq_id'] else '',
        rec="_rec-" + outputs['rec_id'] if outputs['rec_id'] else '',
        run="_run-" + outputs['run_id'] if outputs['run_id'] else ''
    )
    title = '{session}{task}{acq}{rec}{run}'.format(
        session=" Session: " + outputs['session_id'] if outputs['session_id'] else '',
        task=" Task: " + outputs['task_id'] if outputs['task_id'] else '',
        acq=" Acquisition: " + outputs['acq_id'] if outputs['acq_id'] else '',
        rec=" Reconstruction: " + outputs['rec_id'] if outputs['rec_id'] else '',
        run=" Run: " + outputs['run_id'] if outputs['run_id'] else ''
    )
    return name.strip('_'), title


def run_reports(reportlets_dir, out_dir, subject_label, run_uuid):
    """
    Runs the reports

    >>> import os
    >>> from shutil import copytree
    >>> from tempfile import TemporaryDirectory
    >>> filepath = os.path.dirname(os.path.realpath(__file__))
    >>> test_data_path = os.path.realpath(os.path.join(filepath,
    ...                                   '../data/tests/work'))
    >>> curdir = os.getcwd()
    >>> tmpdir = TemporaryDirectory()
    >>> os.chdir(tmpdir.name)
    >>> data_dir = copytree(test_data_path, os.path.abspath('work'))
    >>> os.makedirs('out/fmriprep', exist_ok=True)
    >>> run_reports(os.path.abspath('work/reportlets'),
    ...             os.path.abspath('out'),
    ...             '01', 'madeoutuuid')
    0
    >>> os.chdir(curdir)
    >>> tmpdir.cleanup()

    """
    reportlet_path = os.path.join(reportlets_dir, 'fmriprep', "sub-" + subject_label)
    config = pkgrf('fmriprep', 'viz/config.json')

    out_filename = 'sub-{}.html'.format(subject_label)
    report = Report(reportlet_path, config, out_dir, run_uuid, out_filename)
    return report.generate_report()


def generate_reports(subject_list, output_dir, work_dir, run_uuid):
    """
    A wrapper to run_reports on a given ``subject_list``
    """
    reports_dir = os.path.join(work_dir, 'reportlets')
    report_errors = [
        run_reports(reports_dir, output_dir, subject_label, run_uuid=run_uuid)
        for subject_label in subject_list
    ]

    errno = sum(report_errors)
    if errno:
        import logging
        logger = logging.getLogger('cli')
        logger.warning(
            'Errors occurred while generating reports for participants: %s.',
            ', '.join(['%s (%d)' % (subid, err)
                       for subid, err in zip(subject_list, report_errors)]))
    return errno
