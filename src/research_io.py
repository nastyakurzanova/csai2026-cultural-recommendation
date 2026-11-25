"""Atomic publication of experiment tables with an immediate readback check."""
from pathlib import Path
import os
import pandas as pd

def atomic_csv(frame,path):
    path=Path(path);temp=path.with_name('.'+path.name+'.writing')
    with temp.open('w',encoding='utf-8',newline='') as f:
        frame.to_csv(f,index=False);f.flush();os.fsync(f.fileno())
    # Validate the actual bytes before exposing the new table to the next stage.
    check=pd.read_csv(temp)
    if len(check)!=len(frame) or list(check.columns)!=list(frame.columns):
        raise IOError(f'Incomplete table write: {path.name}')
    os.replace(temp,path)
