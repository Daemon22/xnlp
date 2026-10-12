"""Check PID 17300 and all Python processes."""
import psutil

try:
    p = psutil.Process(17300)
    print("PID 17300: {} status={} cpu={}%".format(p.name(), p.status(), p.cpu_percent()))
    cmd = " ".join(p.cmdline()[:5])
    print("  cmdline:", cmd)
except Exception as e:
    print("PID 17300:", e)

print()
print("All Python processes:")
for p in psutil.process_iter(["pid", "name", "cmdline"]):
    try:
        if p.info["name"] and "python" in p.info["name"].lower():
            cmd = " ".join(p.info["cmdline"] or [])
            print("  PID {}: {}...".format(p.info["pid"], cmd[:120]))
    except:
        pass
