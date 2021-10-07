import os


def check_filename(filename, is_dir=False):
    if is_dir:
        isfile_fnc = os.path.isdir
    else:
        isfile_fnc = os.path.isfile

    if isfile_fnc(filename):
        return filename
    full_path = os.path.abspath(os.getcwd())

    new_filename = os.path.join(full_path, filename)
    if isfile_fnc(new_filename):
        return new_filename

    new_filename = os.path.join("../..", filename)
    if isfile_fnc(new_filename):
        return new_filename

    new_filename = os.path.join(full_path, "../..", filename)
    if isfile_fnc(new_filename):
        return new_filename

    new_filename = os.path.join("datasets", filename)
    if os.path.isdir(new_filename):
        return new_filename

    new_filename = os.path.join(full_path, "datasets", filename)
    if os.path.isdir(new_filename):
        return new_filename

    new_filename = os.path.join("../../datasets", filename)
    if os.path.isdir(new_filename):
        return new_filename

    new_filename = os.path.join(full_path, "../../datasets", filename)
    if os.path.isdir(new_filename):
        return new_filename

    return None
