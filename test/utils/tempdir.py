import shutil
import tempfile
import os

def execute_in_temporary_directory(test_method):
    ''' decorator for tests to run in temporary directories. Creates a temporary
    directory, moves the pwd to that directory, then after the test is run,
    moves the pwd back to the original directory and deletes the temporary
    directory '''

    temporary_directory = tempfile.mkdtemp()
    original_directory = os.getcwd()

    try:
        os.chdir(temporary_directory)
        test_method()
    finally:
        os.chdir(original_directory)
        shutil.rmtree(temporary_directory)
