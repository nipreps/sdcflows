"""Get sdcflows version."""

import sdcflows

with open('/tmp/.docker-version.txt', 'w') as f:
    f.write(sdcflows.__version__)
