"""Bounded passive observation of an already running trainer; never writes game memory."""
import json
import time
from pathlib import Path
import frida

JS = r'''
const base=Process.getModuleByName('SuperMeatBoy.exe').base;
const img=a=>base.add(a-0x400000);
const chars=new Map(); let map=null;
Interceptor.attach(img(0x519d40),{onEnter(){
  const c=this.context.ecx;
  try {if(c.add(0x9e0).readU32()===2)chars.set(c.toString(),c);}catch(e){}
}});
Interceptor.attach(img(0x465b00),{onEnter(){map=this.context.ecx;}});
rpc.exports={sample(){
  const states=[];
  for(const [id,c] of chars){try{states.push({id,s:[0x84,0x88,0x9c,0xa0].map(o=>c.add(o).readFloat()),
    flags:c.add(0x7b4).readU32(),state:c.add(0x7b0).readU32()});}catch(e){}}
  return states;
},info(){
  if(!map)return null;
  const W=map.add(0x20).readU16(),H=map.add(0x22).readU16();
  if(W<4||W>2000||H<4||H>2000)return null;
  const world=img(0x7035c4).readPointer(),objects=[];
  for(let i=0;i<4000;i++){
    const p=world.add(0x124+8*i).readPointer();if(p.isNull())break;
    objects.push([p.add(0x10).readU16()&255,p.add(0x14).readFloat(),p.add(0x18).readFloat()]);
  }
  return {W,H,ox:map.add(0x18).readFloat(),oy:map.add(0x1c).readFloat(),
    words:Array.from(new Uint32Array(map.readPointer().readByteArray(W*H*4))),objects};
}};
'''

root=Path(__file__).resolve().parent
device=frida.get_local_device()
game=next(p for p in device.enumerate_processes() if p.name.lower()=='supermeatboy.exe')
session=device.attach(game.pid)
script=session.create_script(JS)
try:
    script.load()
    rows=[]
    start=time.monotonic()
    while time.monotonic()-start<25:
        rows.append(dict(wall_seconds=time.monotonic()-start,states=script.exports_sync.sample()))
        time.sleep(.25)
    info=script.exports_sync.info()
    (root/'passive_sample.json').write_text(json.dumps(dict(rows=rows,info=info)),encoding='utf-8')
    print('samples',len(rows),'characters',max(len(r['states']) for r in rows),'map',None if info is None else [info['W'],info['H']])
finally:
    script.unload()
    session.detach()
