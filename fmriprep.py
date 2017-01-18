#!/usr/bin/env python3
import sys
import os
import argparse
import subprocess


def main(cmd, *argv):
    parser = argparse.ArgumentParser()

    # Standard FMRIPREP arguments
    parser.add_argument('bids_dir', nargs='?', type=str, default=os.getcwd())
    parser.add_argument('output_dir', nargs='?', type=str,
                        default=os.path.join(os.getcwd(), 'out'))
    parser.add_argument('analysis_level', nargs='?', choices=['participant'],
                        default='participant')

    # Allow alternative images (semi-developer)
    parser.add_argument('-i', '--image', type=str,
                        default='poldracklab/fmriprep:latest')

    # Developer patch/shell options
    parser.add_argument('-f', '--patch-fmriprep', type=str)
    parser.add_argument('-n', '--patch-niworkflows', type=str)
    parser.add_argument('-p', '--patch-nipype', type=str)
    parser.add_argument('--shell', action='store_true')

    # Capture additional arguments to pass inside container
    opts, unknown_args = parser.parse_known_args(argv)

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

    command.extend(['-v', ':'.join((opts.output_dir, '/out')),
                    '-v', ':'.join((opts.bids_dir, '/data', 'ro'))])

    if opts.shell:
        command.append('--entrypoint=bash')

    command.append(opts.image)

    if not opts.shell:
        command.extend(['/data', '/out', opts.analysis_level])
        command.extend(unknown_args)

    print("RUNNING: " + ' '.join(command))
    ret = subprocess.run(command)
    return ret.returncode


if __name__ == '__main__':
    sys.exit(main(*sys.argv))
