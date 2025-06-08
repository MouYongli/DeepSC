from pathlib import Path


def path_of_file(file_path, file_name):
    if file_name == "cell":
        searchKey1 = "cell"
        searchKey2 = ".csv"

    if file_name == "gene":
        searchKey1 = "gene"
        searchKey2 = ".txt"

    files_in_directory = {
        f.name.lower(): f.name for f in file_path.iterdir() if f.is_file()
    }
    lower_files = list(files_in_directory.keys())
    search_file_path = Path("")

    search_files = [
        f for f in lower_files if f.startswith(searchKey1) and f.endswith(searchKey2)
    ]
    if search_files:
        if not len(search_files) > 1:
            # print(f"find {file_name} file: {search_files[0]} in path {file_path}")
            original_file_name = files_in_directory[search_files[0]]
            search_file_path = file_path / original_file_name
            return search_file_path
        else:
            print(f"Multiple files found in path {file_path}")
    else:
        parent_folder = file_path.parent
        files_in_parent_directory = {
            f.name.lower(): f.name for f in parent_folder.iterdir() if f.is_file()
        }
        lower_files_in_parent_directory = list(files_in_parent_directory.keys())
        search_files = [
            f
            for f in lower_files_in_parent_directory
            if f.startswith(searchKey1) and f.endswith(searchKey2)
        ]
        if search_files:
            if not len(search_files) > 1:
                original_file_name = files_in_parent_directory[search_files[0]]
                search_file_path = parent_folder / original_file_name
                # print(f"find gene file: {search_files[0]} in path {parent_folder}")
                return search_file_path
            else:
                print(f"Multiple files found in path {file_path}")
        else:
            print(f"Corresponding file not found in path {file_path}")
