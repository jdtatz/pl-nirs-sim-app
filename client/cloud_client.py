import os
import time
import json
import subprocess
from string import Template
from pfurl.pfurl import Pfurl

proc_id = "100"
host_ip = subprocess.check_output("ip route | grep -v docker | awk '{if(NF==11) printf $9}'", shell=True, encoding ='UTF-8')

with open('cloud_proc_run.json', 'r') as f:
    proc_run_tmpl = Template(f.read())

with open('cloud_proc_check.json', 'r') as f:
    proc_check_tmpl = Template(f.read())

default = {
    "verb": "POST",
    "http": host_ip + ":5005/api/v1/cmd",
    "contentType": "",
    "auth": "chris:chris1234",
    "b_raw": True,
    "b_quiet": True,
    "b_oneShot": False,
    "b_httpResponseBodyParse": True,
    "jsonwrapper": "payload",
    "startFromCLI": False
}

proc_ids = {}
for wave in np.linspace(650, 1000, 100, dtype=np.int32):
    pid = "nirs_sim_test_wave_%d" % wave
    msg = json.loads(proc_run_tmpl.safe_substitute(proc_id=pid, service="host", spec_file="spec.xz", wavelength=wave))
    pf = Pfurl(**default, msg=msg)
    res = json.loads(pf())  # TODO: Check if succsses
    print("Start:", wave, pid, res)
    proc_ids[pid] = proc_check_tmpl.safe_substitute(proc_id=pid)

while len(proc_ids):
    pid, msg = proc_ids.popitem()
    pf = Pfurl(**default, msg=msg)
    res = json.loads(pf())  # TODO: Check if succsses
    print("Check:", pid, res)
    if not res['status']:  # not done
        proc_ids[pid] = msg
        time.sleep(5)

# Process output
# results = np.load("~/nirs/out/{}/out.npz".format(pid))