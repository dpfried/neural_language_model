#!/usr/bin/env python
import subprocess

sets_by_folder = {
    "Training": "10a 1a 2c 2h 3a 3c 4c 5d 5i 7a".split(),
    "Testing": "1b 1c 1d 1e 2a 2b 2d 2e 2f 2g 2h 2i 2j 3b 3d 3e 3f 3g 3h 4a 4b 4d 4e 4f 4g 4h 5a 5b 5c 5e 5f 5g 5h 6a 6b 6c 6d 6e 6f 6g 6h 7b 7c 7d 7e 7f 7g 7h 8a 8b 8c 8d 8e 8f 8g 8h 9a 9b 9c 9d 9e 9f 9g 9h 9i 10b 10c 10d 10e 10f".split()
}

for folder, sets_to_run in sets_by_folder.items():
    for s in sets_to_run:
        subprocess.call(["test_mine.sh", folder, s])
