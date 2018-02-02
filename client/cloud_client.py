import os
import time
import json
import subprocess
import numpy as np
from pfurl.pfurl import Pfurl

proc_id = "100"
host_ip = subprocess.check_output("ip route | grep -v docker | awk '{if(NF==11) printf $9}'", shell=True, encoding ='UTF-8')


def create_cloud_proc_run(proc_id, service, spec, wave):
    return json.dumps({
        "action": "coordinate",
        "threadAction": True,
        "meta-store": {
            "meta": "meta-compute",
            "key": "jid"
        },
        "meta-data": {
            "remote": {
                "key": "%meta-store"
            },
            "localSource": {
                "path": "~/nirs/in"
            },
            "localTarget": {
                "path": "~/nirs/out",
                "createDir": False
            },
            "specialHandling": {
                "op": "plugin",
                "cleanup": True
            },
            "transport": {
                "mechanism": "compress",
                "compress": {
                    "encoding": "none",
                    "archive": "zip",
                    "unpack": True,
                    "cleanup": True
                }
            },
            "service": service
        },
        "meta-compute": {
            "cmd": "$execshell $selfpath/$selfexec /share/incoming /share/outgoing --spec={} --wavelength={}".format(spec, wave),
            "auid": "jacob",
            "jid": proc_id,
            "threaded": True,
            "container": {
                "target": {
                    "image": "jdtatz/pl-nirs-sim-app",
                    "cmdParse": True
                },
                "manager": {
                    "image": "fnndsc/swarm",
                    "app": "swarm.py",
                    "env": {
                        "meta-store": "key",
                        "serviceType": "docker",
                        "shareDir": "%shareDir",
                        "serviceName": proc_id
                    }
                }
            },
            "service": service
        }
    })


def create_cloud_proc_check(proc_id):
    return json.dumps({
        "action": "status",
        "threadAction": False,
        "meta": {
            "remote": {
                "key": proc_id
            }
        }
    })

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
for wavelength in np.linspace(650, 1000, 100, dtype=np.int32):
    pid = "nirs_sim_test_wave_%d" % wavelength
    msg = create_cloud_proc_run(pid, "host", "spec.xz", wavelength)
    pf = Pfurl(**default, msg=msg)
    res = pf()
    if res:
        result = json.loads(res)
        if(result['status']):
            print("Start:", wavelength, pid, res)
            proc_ids[pid] = create_cloud_proc_check(pid)
        else:
            print("Failed to run:", pid, res, result)
    else:
        print("Failed to connect:", pid, res)

while len(proc_ids):
    pid, msg = proc_ids.popitem()
    pf = Pfurl(**default, msg=msg)
    res = pf()
    if res:
        result = json.loads(res)
        if result['status']:
            print("Finished:", pid, res, result)
            continue
        else:
            print("Still going:", pid, res, result)
    else:
        print("Failed to connect:", pid, res)
    time.sleep(10)
    proc_ids[pid] = msg

# Process output
# results = np.load("~/nirs/out/{}/out.npz".format(pid))