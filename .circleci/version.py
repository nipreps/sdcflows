"""Get sdcflows version."""
import sdcflows

print(sdcflows.__version__, end="", file=open("/tmp/.docker-version.txt", "w"))
