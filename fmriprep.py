#!/usr/bin/env python3
import sys
import os
import re
import argparse
import subprocess

__version__ = 0.1
__bugreports__ = 'https://github.com/poldracklab/fmriprep/issues'
MISSING = """
Image '{}' is missing
Would you like to download? [Y/n] """


def check_docker():
    """Verify that docker is installed and the user has permission to
    run docker images.

    Returns
    -------
    -1  Docker can't be found
     0  Docker found, but user can't connect to daemon
     1  Test run OK
     """
    try:
        ret = subprocess.run(['docker', 'version'], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
    except FileNotFoundError as e:
        return -1
    if ret.stderr.startswith(b"Cannot connect to the Docker daemon."):
        return 0
    return 1


def check_image(image):
    """Check whether image is present on local system"""
    ret = subprocess.run(['docker', 'images', '-q', image],
                         stdout=subprocess.PIPE)
    return bool(ret.stdout)


def merge_help(wrapper_help, target_help):
    # Matches all flags with up to one nested square bracket
    opt_re = re.compile(r'(\[--?[\w-]+(?:[^\[\]]+(?:\[[^\[\]]+\])?)?\])')
    # Matches flag name only
    flag_re = re.compile(r'\[--?([\w-]+)[ \]]')

    # Normalize to Unix-style line breaks
    w_help = wrapper_help.rstrip().replace('\r', '')
    t_help = target_help.rstrip().replace('\r', '')

    w_usage, *w_groups = w_help.split('\n\n')
    t_usage, *t_groups = t_help.split('\n\n')

    w_posargs = w_usage.split('\n')[-1].lstrip()
    t_posargs = t_usage.split('\n')[-1].lstrip()

    w_options = opt_re.findall(w_usage)
    w_flags = sum(map(flag_re.findall, w_options), [])
    t_options = opt_re.findall(t_usage)
    t_flags = sum(map(flag_re.findall, t_options), [])

    # The following code makes this assumption
    assert w_flags[:2] == ['h', 'v']
    assert w_posargs.replace(']', '').replace('[', '') == t_posargs

    # Make sure we're not clobbering options we don't mean to
    overlap = set(w_flags).intersection(t_flags)
    assert overlap == set('hv'), "Clobbering options: {}".format(
        ', '.join(overlap - set('hv')))

    sections = []

    # Construct usage
    start = w_usage[:w_usage.index(' [')]
    indent = ' ' * len(start)
    new_options = sum((
        w_options[:2],
        [opt for opt, flag in zip(t_options, t_flags) if flag not in overlap],
        w_options[2:]
        ), [])
    opt_line_length = 79 - len(start)
    length = 0
    opt_lines = [start]
    for opt in new_options:
        opt = ' ' + opt
        olen = len(opt)
        if length + olen <= opt_line_length:
            opt_lines[-1] += opt
            length += olen
        else:
            opt_lines.append(indent + opt)
            length = olen
    opt_lines.append(indent + ' ' + t_posargs)
    sections.append('\n'.join(opt_lines))

    # Use target description and positional args
    sections.extend(t_groups[:2])

    for line in t_groups[2].split('\n')[1:]:
        content = line.lstrip().split(',', 1)[0]
        if content[1:] not in overlap:
            w_groups[2] += '\n' + line

    sections.append(w_groups[2])

    # All remaining sections, show target then wrapper
    sections.extend(t_groups[3:] + w_groups[3:])
    return '\n\n'.join(sections)


def main(cmd, *argv):
    parser = argparse.ArgumentParser(
        description='fMRI Preprocessing workflow',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False)

    # Standard FMRIPREP arguments
    parser.add_argument('bids_dir', nargs='?', type=str, default=os.getcwd())
    parser.add_argument('output_dir', nargs='?', type=str,
                        default=os.path.join(os.getcwd(), 'out'))
    parser.add_argument('analysis_level', nargs='?', choices=['participant'],
                        default='participant')

    parser.add_argument('-h', '--help', action='store_true',
                        help="show this help message and exit")
    parser.add_argument('-v', '--version', action='store_true',
                        help="show program's version number and exit")

    # Allow alternative images (semi-developer)
    parser.add_argument('-i', '--image', metavar='IMG', type=str,
                        default='poldracklab/fmriprep:latest',
                        help='image name')

    # Developer patch/shell options
    g_dev = parser.add_argument_group(
        'Developer options',
        'Tools for testing and debugging FMRIPREP')
    g_dev.add_argument('-f', '--patch-fmriprep', metavar='PATH',
                       type=os.path.abspath,
                       help='working fmriprep repository')
    g_dev.add_argument('-n', '--patch-niworkflows', metavar='PATH',
                       type=os.path.abspath,
                       help='working niworkflows repository')
    g_dev.add_argument('-p', '--patch-nipype', metavar='PATH',
                       type=os.path.abspath,
                       help='working nipype repository')
    g_dev.add_argument('--shell', action='store_true',
                       help='open shell in image instead of running FMRIPREP')

    # Capture additional arguments to pass inside container
    opts, unknown_args = parser.parse_known_args(argv)

    # Stop if no docker / docker fails to run
    check = check_docker()
    if check < 1:
        if opts.version:
            print('fmriprep wrapper {!s}'.format(__version__))
        if opts.help:
            parser.print_help()
        print("fmriprep: ", end='')
        if check == -1:
            print("Could not find docker command... Is it installed?")
        else:
            print("Make sure you have permission to run 'docker'")
        return 1

    # For --help or --version, ask before downloading an image
    if not check_image(opts.image):
        resp = 'Y'
        if opts.version:
            print('fmriprep wrapper {!s}'.format(__version__))
        if opts.help:
            parser.print_help()
        if opts.version or opts.help:
            try:
                resp = input(MISSING.format(opts.image))
            except KeyboardInterrupt:
                print()
                return 1
        if resp not in ('y', 'Y', ''):
            return 0

    command = ['docker', 'run', '--rm', '-it']

    if opts.patch_fmriprep is not None:
        command.extend(['-v', ':'.join((opts.patch_fmriprep,
                                        '/root/src/fmriprep'))])
    if opts.patch_niworkflows:
        command.extend(['-v', ':'.join((opts.patch_niworfklows,
                                        '/root/src/niworkflows'))])
    if opts.patch_nipype:
        command.extend(['-v', ':'.join((opts.patch_nipype,
                                        '/root/src/nipype'))])

    command.extend(['-v', ':'.join((opts.bids_dir, '/data', 'ro')),
                    '-v', ':'.join((opts.output_dir, '/out')),
                    ])

    if opts.shell:
        command.append('--entrypoint=bash')

    command.append(opts.image)

    # Override help and version to describe underlying program
    # Respects '-i' flag, so will retrieve information from any image
    if opts.help:
        command.append('-h')
        targethelp = subprocess.check_output(command).decode()
        print(merge_help(parser.format_help(), targethelp))
        return 0
    elif opts.version:
        # Get version to be run and exit
        command.append('-v')
        ret = subprocess.run(command)
        return ret.returncode

    if not opts.shell:
        command.extend(['/data', '/out', opts.analysis_level])
        command.extend(unknown_args)

    print("RUNNING: " + ' '.join(command))
    ret = subprocess.run(command)
    if ret.returncode:
        print("fmriprep: Please report errors to {}".format(__bugreports__))
    return ret.returncode


if __name__ == '__main__':
    sys.exit(main(*sys.argv))
