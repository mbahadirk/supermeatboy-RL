"""Native A+ progression helpers for PPO only; no action selection or learning code."""
import math
import os
import shutil
import subprocess
import time
from levels_smb import a_plus_time


class APlusProgress:
    """Keep the best real finish and count five completed waves after the first A+."""
    def __init__(self):
        self.first_wave=None
        self.last_wave=-1
        self.best=None

    def observe(self,level,message):
        wave=message['wave']
        if wave<=self.last_wave:return None
        self.last_wave=wave
        candidate=aplus_candidate(level,message)
        if candidate is not None:
            if self.first_wave is None:self.first_wave=wave
            if self.best is None or candidate['seconds']<self.best['seconds']:self.best=candidate
        if self.best is None or wave<self.first_wave+5:return None
        return dict(self.best,first_wave=self.first_wave,after_wave=wave)


def aplus_candidate(level,message):
    """Grade native HUD seconds, never frame/60 in the variable-step runner."""
    par=a_plus_time(level)
    if par is None:return None
    seconds=message.get('finish_seconds') or [];frames=message.get('finish') or []
    out=[]
    for i,status in enumerate(message.get('status') or []):
        if status!=2 or i>=len(seconds) or i>=len(frames) or frames[i]<0:continue
        t=seconds[i]
        if isinstance(t,bool) or not isinstance(t,(int,float)) or not math.isfinite(t) or not 0<t<=par:continue
        out.append(dict(level=level,wave=message['wave'],agent=i,seconds=float(t),par=float(par)))
    return min(out,key=lambda c:(c['seconds'],c['agent'])) if out else None


def locate_save():
    result=subprocess.run(['powershell','-NoProfile','-Command',
        'Get-CimInstance Win32_Process -Filter "Name=\'SuperMeatBoy.exe\'" | ForEach-Object { $_.ExecutablePath }'],
        capture_output=True,text=True,timeout=20)
    paths=[p.strip() for p in result.stdout.splitlines() if p.strip()]
    if not paths:raise FileNotFoundError('the running SuperMeatBoy.exe path was not found')
    save=os.path.join(os.path.dirname(paths[0]),'UserData','savegame.dat')
    if not os.path.isfile(save):raise FileNotFoundError(save)
    return save


def backup_save(save,root):
    stamp=time.strftime('save_%Y%m%d-%H%M%S');dest=os.path.join(root,stamp);index=1
    while os.path.exists(dest):
        index+=1;dest=os.path.join(root,f'{stamp}_{index}')
    os.makedirs(dest);shutil.copy2(save,os.path.join(dest,os.path.basename(save)))
    return dest
