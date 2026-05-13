import os
import time

while True:
    os.system("git add .")
    os.system('git commit -m "Auto update"')
    os.system("git push origin main")

    print("Changes pushed.")
    
    time.sleep(300)  # every 5 minutes