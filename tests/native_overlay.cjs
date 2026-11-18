// Exercise delayed D3D device availability at the native boundary, never a live game.
const assert = require('node:assert/strict');
const vm = require('node:vm');
const {context, heap, Mem, sent, hooks, functions, receivers} = require('./native_harness.cjs')();
assert.equal(vm.runInContext('pathPresentHook', context), null); // no graphics device at script load
vm.runInContext('active = true; ready = true;', context);
receivers.get('path')({payload:[{pts:[[0,0],[10,0]],col:0xff00ff00,w:3}]});
// Graphics/device/vtable become available later. Supply real return types for the D3D calls in use.
const gfx=0x300000, device=0x310000, vt=0x320000;
heap.set(0x709068,gfx); heap.set(gfx,device); heap.set(device,vt);
for(let i=0;i<120;i++) heap.set(vt+4*i,0x400000+i*16);
const address=i=>0x400000+i*16;
functions.set(address(48),(_d,p)=>{[0,0,1280,720,0,1].forEach((v,i)=>p.add(i*4).writeU32(v));return 0;});
for(const i of [58,64,66,88,90,93,101,108]) functions.set(address(i),(...a)=>{
  a[a.length-1].writeU32(0);return 0;
});
functions.set(address(38),(_d,_i,p)=>{p.writePointer(new Mem(0x510000));return 0;});
functions.set(address(18),(_d,_a,_b,_c,p)=>{p.writePointer(new Mem(0x520000));return 0;});
functions.set(address(41),()=>0);
let draws=0;
functions.set(address(83),(_d,type,count,_vertices,stride)=>{
  assert.equal(type,4);assert.equal(stride,20);assert.ok(count>0);draws++;return 0;
});
// No scene-layer hook fires, just as in the measured later-level render path.
receivers.get('path')({payload:[{pts:[[0,0],[10,0]],col:0xff00ff00,w:3}]});
const livePresent=()=>hooks.filter(h=>h.address.address===address(17)&&!h.detached);
assert.equal(livePresent().length,1);
receivers.get('path')({payload:[{pts:[[0,0],[10,0]],col:0xff00ff00,w:3}]});
assert.equal(livePresent().length,1);
const camera=hooks.find(h=>h.address.address===address(94)&&!h.detached);
const matrix=new Mem(0x530000);
const M=[1.357995,0,0,291.0533, 0,2.414213,0,455.5594, 0,0,1.0001,499.0499, 0,0,1,500];
M.forEach((v,i)=>matrix.add(4*i).writeFloat(v)); // first world matrix measured by overlay_probe.py
camera.callbacks.onEnter([new Mem(device),new Mem(0),matrix,new Mem(4)]);
assert.equal(vm.runInContext('camM[0]',context),M[0]);
matrix.writeFloat(40739.84); // later per-object matrices must not replace this frame's world camera
camera.callbacks.onEnter([new Mem(device),new Mem(0),matrix,new Mem(4)]);
assert.equal(vm.runInContext('camM[0]',context),M[0]);
livePresent()[0].callbacks.onEnter([new Mem(device)]);
assert.equal(draws,1);
assert.deepEqual(sent.filter(m=>m.ev==='overlay').map(m=>m.stage),
                 ['paths received','device hook ready','camera ready','drawn']);
// A device with a changed Present implementation replaces its hook; another device cannot consume freshness.
heap.set(vt+4*17,0x600000);
const device2=device+0x100;
heap.set(gfx,device2);heap.set(device2,vt);
receivers.get('path')({payload:[{pts:[[0,0],[10,0]],col:0xff00ff00,w:3}]});
assert.equal(hooks.filter(h=>h.address.address===address(17)&&!h.detached).length,0);
const replacement=hooks.find(h=>h.address.address===0x600000&&!h.detached);
const nextCamera=hooks.find(h=>h.address.address===address(94)&&!h.detached);
M.forEach((v,i)=>matrix.add(4*i).writeFloat(v));
nextCamera.callbacks.onEnter([new Mem(device2),new Mem(0),matrix,new Mem(4)]);
replacement.callbacks.onEnter([new Mem(123)]);assert.equal(draws,1);
replacement.callbacks.onEnter([new Mem(device2)]);assert.equal(draws,2);
console.log('native overlay late-device, draw and hook replacement checks passed');
