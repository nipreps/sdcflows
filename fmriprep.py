#!/usr/bin/env python3
import sys
import os
import argparse
import subprocess

code_base = '/data/chris/Projects/poldracklab'
data_base = '/data/fmriprep/data'
working = '/data/fmriprep/working'


def main(cmd, *argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--patch_fmriprep', action='store_true')
    parser.add_argument('-w', '--patch_niworkflows', action='store_true')
    parser.add_argument('-n', '--patch_nipype', action='store_true')
    parser.add_argument('--fmriprep_path', type=str,
                        default=os.path.join(code_base, 'fmriprep'))
    parser.add_argument('--niworkflows_path', type=str,
                        default=os.path.join(code_base, 'niworkflows'))
    parser.add_argument('--nipype_path', type=str,
                        default=os.path.join(code_base, 'nipype'))
    parser.add_argument('--data_path', type=str,
                        default=os.path.join(data_base, 'ds005'))
    parser.add_argument('--out_path', type=str,
                        default=os.path.join(working, 'out'))
    parser.add_argument('--shell', action='store_true')
    parser.add_argument('-i', '--image', type=str,
                        default='poldracklab/fmriprep:latest')
    parser.add_argument('ARGS', type=str, nargs='*')
    opts = parser.parse_args(argv)

    command = ['docker', 'run', '--rm', '-it']

    if opts.patch_fmriprep:
        command.extend(['-v', ':'.join((opts.fmriprep_path,
                                        '/root/src/fmriprep', 'ro'))])
    if opts.patch_niworkflows:
        command.extend(['-v', ':'.join((opts.niworfklows_path,
                                        '/root/src/niworkflows', 'ro'))])
    if opts.patch_nipype:
        command.extend(['-v', ':'.join((opts.nipype_path,
                                        '/root/src/nipype', 'ro'))])

    command.extend(['-v', ':'.join((opts.out_path, '/out')),
                    '-v', ':'.join((opts.data_path, '/data', 'ro'))])

    if opts.shell:
        command.append('--entrypoint=bash')

    command.append(opts.image)
    ret = subprocess.run(command + opts.ARGS)
    return ret.returncode

if __name__ == '__main__':
    sys.exit(main(*sys.argv))
