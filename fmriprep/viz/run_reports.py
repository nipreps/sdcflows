from pkg_resources import resource_filename as pkgrf

from reports import Report

if __name__ == '__main__':
    path = pkgrf('fmriprep', '../out/images/')
    out_dir = pkgrf('fmriprep', '../out/')
    config = pkgrf('fmriprep', 'viz/config.json')
    report = Report(path, config, out_dir)
    for sub_report in report.sub_reports:
        for element in sub_report.elements:
            print(element.files)
            print(element.title)
    print(report.generate_report())
