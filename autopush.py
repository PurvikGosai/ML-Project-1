import os
import time

while True:
    # Check if there are changes
    status = os.popen("git status --porcelain").read().strip()

    if status:
        print("Changes detected...")

        os.system("git add .")
        os.system('git commit -m "Auto update"')
        os.system("git push origin main")

        print("Changes pushed successfully.")

    else:
        print("No changes detected.")

    time.sleep(300)   # checks every 5 minutes