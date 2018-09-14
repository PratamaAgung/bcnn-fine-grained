import os
import glob

for filename in glob.glob("data_gemastik/*/*"):
    new_filename = filename.replace(" ", "")
    os.rename(filename, new_filename)
