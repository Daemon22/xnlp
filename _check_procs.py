"""Check running XNLP processes."""
import psutil

for p in psutil.process_iter(['pid', 'name', 'cmdline']):
    try:
        cmd = ' '.join(p.info['cmdline'] or [])
        if 'watch_and_eval' in cmd or 'train_v2' in cmd or 'xnlp_trainer' in cmd:
            print(f"PID {p.info['pid']}: {cmd[:150]}")
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
